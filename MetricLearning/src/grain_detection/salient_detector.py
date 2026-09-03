"""
SalientObjectDetector — Wrapper U²-Net para detección de objetos salientes.

Adaptado de XYZ_Ctrl_L206_GUI/src/ai_segmentation.py para el proyecto MetricLearning.
Genera mapas de saliencia pixel-level que permiten localizar granos de polen
sin necesidad de anotaciones de bounding box.

Ref: Qin et al. (2020) U²-Net: Going Deeper with Nested U-Structure for SOD
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.models.u2net import U2NET, U2NETP

logger = logging.getLogger(__name__)

WEIGHTS_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "weights"
U2NET_WEIGHTS = WEIGHTS_DIR / "u2net.pth"
U2NETP_WEIGHTS = WEIGHTS_DIR / "u2netp.pth"

WEIGHTS_URLS = {
    "u2net": "https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ",
    "u2netp": "https://drive.google.com/uc?id=1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy",
}


class SalientObjectDetector:
    """
    Detector de Objetos Salientes usando U²-Net.

    Permite aislar granos de polen del fondo de microscopía usando
    deep learning, sin necesidad de calibración ni anotación previa.

    Attributes:
        model_type: 'u2netp' (rápido, ~4MB) o 'u2net' (preciso, ~176MB)
        device: 'cuda' o 'cpu'
        input_size: Tamaño de entrada del modelo (default 320x320)
    """

    def __init__(
        self,
        model_type: str = "u2netp",
        device: Optional[str] = None,
        input_size: int = 320,
        auto_download: bool = True,
    ):
        self.model_type = model_type
        self.input_size = input_size
        self.auto_download = auto_download

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"[SalientObjectDetector] Dispositivo: {self.device}")

        self.model = None
        self._load_model()

    def _get_weights_path(self) -> Path:
        if self.model_type == "u2net":
            return U2NET_WEIGHTS
        return U2NETP_WEIGHTS

    def _download_weights(self) -> bool:
        weights_path = self._get_weights_path()

        if weights_path.exists():
            return True

        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

        url = WEIGHTS_URLS.get(self.model_type)
        if not url:
            logger.error(f"[SalientObjectDetector] URL no encontrada para {self.model_type}")
            return False

        logger.info(f"[SalientObjectDetector] Descargando pesos {self.model_type}...")

        try:
            import gdown

            gdown.download(url, str(weights_path), quiet=False)
            logger.info(f"[SalientObjectDetector] Pesos descargados: {weights_path}")
            return True
        except ImportError:
            logger.error(
                "[SalientObjectDetector] gdown no instalado. Ejecuta: pip install gdown"
            )
            logger.error(f"[SalientObjectDetector] O descarga manualmente desde: {url}")
            logger.error(f"[SalientObjectDetector] Guardar en: {weights_path}")
            return False
        except Exception as e:
            logger.error(f"[SalientObjectDetector] Error descargando pesos: {e}")
            return False

    def _load_model(self):
        weights_path = self._get_weights_path()

        if not weights_path.exists():
            if self.auto_download:
                if not self._download_weights():
                    raise FileNotFoundError(
                        f"No se encontraron pesos en {weights_path}. "
                        f"Ejecuta setup_u2net.py o descarga manualmente."
                    )
            else:
                raise FileNotFoundError(
                    f"Pesos no encontrados: {weights_path}. "
                    f"Ejecuta setup_u2net.py para descargarlos."
                )

        if self.model_type == "u2net":
            self.model = U2NET(3, 1)
        else:
            self.model = U2NETP(3, 1)

        logger.info(f"[SalientObjectDetector] Cargando pesos desde {weights_path}")
        state_dict = torch.load(
            str(weights_path), map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        logger.info(f"[SalientObjectDetector] Modelo {self.model_type} cargado exitosamente")

    @staticmethod
    def enhance(image: np.ndarray) -> np.ndarray:
        """CLAHE en L: único preproceso de contraste antes de U²-Net."""
        if image is None or image.size == 0:
            return image
        bgr = image
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        elif bgr.shape[2] == 4:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        span = float(np.percentile(l_ch, 90) - np.percentile(l_ch, 10))
        clip = float(1.0 + 3.0 * (1.0 - min(1.0, span / 255.0)))
        n = max(2, int(min(l_ch.shape) // 16))
        tile = 1 << int(np.log2(n))
        enhanced = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(l_ch)
        logger.debug(
            "[U2-Net] CLAHE clip=%.2f tile=%d  Lσ %.1f → %.1f",
            clip, tile, float(np.std(l_ch)), float(np.std(enhanced)),
        )
        return cv2.cvtColor(cv2.merge([enhanced, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    def _preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
        """
        Preprocesa imagen para el modelo manteniendo aspect ratio mediante padding.

        Args:
            image: Imagen BGR o grayscale (numpy array)

        Returns:
            (tensor, original_size, pad_info)
        """
        image = self.enhance(image)
        original_size = image.shape[:2]
        h, w = original_size

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        longest = max(h, w)
        pad_top = (longest - h) // 2
        pad_bottom = longest - h - pad_top
        pad_left = (longest - w) // 2
        pad_right = longest - w - pad_left
        pad_info = (pad_top, pad_left)

        image_padded = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

        image_resized = cv2.resize(image_padded, (self.input_size, self.input_size))

        image_resized = image_resized.astype(np.float32) / 255.0
        image_resized = (image_resized - np.array([0.485, 0.456, 0.406])) / np.array(
            [0.229, 0.224, 0.225]
        )

        tensor = torch.from_numpy(image_resized.transpose(2, 0, 1)).float().unsqueeze(0)

        return tensor.to(self.device), original_size, pad_info

    def _postprocess(
        self, output: torch.Tensor, original_size: Tuple[int, int], pad_info: Tuple[int, int]
    ) -> np.ndarray:
        """
        Postprocesa la salida del modelo, recortando el padding para restaurar tamaño.

        Args:
            output: Tensor de salida del modelo
            original_size: (H, W) tamaño original
            pad_info: (pad_top, pad_left) usado en preprocess

        Returns:
            Máscara de probabilidad [0-1] en tamaño original
        """
        mask = output[0].squeeze().cpu().numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

        h, w = original_size
        longest = max(h, w)
        pad_top, pad_left = pad_info

        mask_padded = cv2.resize(mask, (longest, longest))
        mask_cropped = mask_padded[pad_top : pad_top + h, pad_left : pad_left + w]

        return mask_cropped

    @torch.no_grad()
    def get_saliency_map(self, image: np.ndarray) -> np.ndarray:
        """
        Obtiene mapa de probabilidad de saliencia [0-1] en tamaño original.

        Args:
            image: Imagen BGR o grayscale (numpy array)

        Returns:
            Mapa de probabilidad float32 [0-1], shape = imagen original
        """
        if self.model is None:
            raise RuntimeError("Modelo no cargado")

        tensor, original_size, pad_info = self._preprocess(image)
        outputs = self.model(tensor)
        mask = self._postprocess(outputs, original_size, pad_info)

        return mask.astype(np.float32)

    @torch.no_grad()
    def get_binary_mask(
        self, image: np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        """
        Obtiene máscara binaria (0/255) de objetos salientes.

        Args:
            image: Imagen BGR o grayscale
            threshold: Umbral de binarización [0-1]

        Returns:
            Máscara binaria uint8 (0 o 255)
        """
        prob_mask = self.get_saliency_map(image)
        return (prob_mask > threshold).astype(np.uint8) * 255

    def is_ready(self) -> bool:
        return self.model is not None

    def get_device(self) -> str:
        return str(self.device)
