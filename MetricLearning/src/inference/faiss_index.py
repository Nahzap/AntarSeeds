"""
FAISS Index wrapper para búsqueda vectorial eficiente.
Soporta IndexFlatL2, IndexFlatIP, e IndexIVFPQ.
Ref: [R11] Douze et al. 2024, [R12] Rahman et al. 2024.
"""

import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss not installed. Install with: pip install faiss-cpu")


class FAISSIndex:
    """
    Wrapper para índices FAISS con soporte para búsqueda exacta y aproximada.
    
    Args:
        dimension: Dimensión de los embeddings
        index_type: Tipo de índice ('flat_l2', 'flat_ip', 'ivfpq')
        metric: 'l2' o 'inner_product'
        nlist: Número de clusters para IVF (default 100)
        nprobe: Número de clusters a buscar (default 10)
        pq_m: Subquantizers para PQ (default 16)
        pq_nbits: Bits por subquantizer (default 8)
        use_gpu: Usar GPU para FAISS (default False)
        
    Example:
        >>> index = FAISSIndex(dimension=128, index_type='flat_ip')
        >>> index.add(embeddings, labels)
        >>> distances, indices, labels = index.search(query, k=5)
    """
    
    def __init__(self, dimension: int, index_type: str = 'flat_ip',
                 metric: str = 'inner_product', nlist: int = 100,
                 nprobe: int = 10, pq_m: int = 16, pq_nbits: int = 8,
                 use_gpu: bool = False):
        
        if not FAISS_AVAILABLE:
            raise ImportError("faiss is required. Install with: pip install faiss-cpu")
        
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.labels = None
        self._index = None
        
        if index_type == 'flat_l2':
            self._index = faiss.IndexFlatL2(dimension)
        elif index_type == 'flat_ip':
            self._index = faiss.IndexFlatIP(dimension)
        elif index_type == 'ivfpq':
            quantizer = faiss.IndexFlatL2(dimension)
            self._index = faiss.IndexIVFPQ(quantizer, dimension, nlist, pq_m, pq_nbits)
            self._index.nprobe = nprobe
        else:
            raise ValueError(f"Unknown index_type: {index_type}. Options: flat_l2, flat_ip, ivfpq")
        
        if use_gpu and hasattr(faiss, 'StandardGpuResources'):
            try:
                res = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(res, 0, self._index)
                logger.info("FAISS index moved to GPU")
            except Exception as e:
                logger.warning(f"GPU FAISS failed, using CPU: {e}")
        
        logger.info(f"FAISSIndex created: type={index_type}, dim={dimension}, metric={metric}")
    
    def add(self, embeddings: np.ndarray, labels: np.ndarray):
        """
        Agrega embeddings al índice.
        
        Args:
            embeddings: (N, D) float32
            labels: (N,) int
        """
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        
        if self.index_type == 'ivfpq' and not self._index.is_trained:
            logger.info(f"Training IVFPQ index with {len(embeddings)} vectors...")
            self._index.train(embeddings)
        
        self._index.add(embeddings)
        self.labels = np.array(labels)
        
        logger.info(f"Added {len(embeddings)} vectors to index (total: {self._index.ntotal})")
    
    def search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Busca los K vecinos más cercanos.
        
        Args:
            query: (N, D) float32 — queries
            k: Número de vecinos
            
        Returns:
            (distances, indices, labels) — labels es None si no se proporcionaron
        """
        query = np.ascontiguousarray(query, dtype=np.float32)
        
        distances, indices = self._index.search(query, k)
        
        result_labels = None
        if self.labels is not None:
            valid_mask = indices >= 0
            result_labels = np.full_like(indices, -1)
            result_labels[valid_mask] = self.labels[indices[valid_mask]]
        
        return distances, indices, result_labels
    
    def reset(self):
        """Limpia el índice."""
        self._index.reset()
        self.labels = None
        logger.info("FAISS index reset")
    
    @property
    def ntotal(self) -> int:
        """Número total de vectores en el índice."""
        return self._index.ntotal
    
    def save(self, path: str):
        """Guarda el índice a disco."""
        faiss.write_index(self._index, path)
        if self.labels is not None:
            np.save(path + '.labels.npy', self.labels)
        logger.info(f"FAISS index saved to {path}")
    
    def load(self, path: str):
        """Carga el índice desde disco."""
        self._index = faiss.read_index(path)
        labels_path = path + '.labels.npy'
        import os
        if os.path.exists(labels_path):
            self.labels = np.load(labels_path)
        logger.info(f"FAISS index loaded from {path} ({self._index.ntotal} vectors)")
