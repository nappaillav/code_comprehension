from typing import List
import torch
import torch.nn.functional as F


class CEWithChunkedOutputLoss(torch.nn.Module):
    """
    CE with chunked outputs that saves memory by only upcasting one chunk at a time.

    Since the model is trained with bf16, before running CE, we have to upcast
    it to fp32 for better accuracy and stability. When upcasting happens, the memory usage doubles.
    Models like llama3 have large vocabulary size and, therefore, have a large output
    result (bsz, num_tokens, vocab_size). If we chunk on the token level, you can still compute
    the cross entropy normally, but upcasting only one chunk at a time saves considerable memory.

    The CE and upcasting have to be compiled together for better performance.
    When using this class, we recommend using torch.compile only on the method `compute_cross_entropy`.
    The gains from chunking won't be realized if you compile the entire class.

    For more details, please refer to: https://github.com/pytorch/torchtune/pull/1390
    """

    def __init__(self, num_output_chunks: int = 8, ignore_index: int = -100):
        super().__init__()
        self.num_output_chunks = num_output_chunks
        self.ignore_index = ignore_index

    def compute_cross_entropy(
        self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Upcast logits to fp32 and compute cross entropy loss.
        """
        # Apply mask to labels
        masked_labels = labels.clone()
        masked_labels[~mask] = self.ignore_index
        
        return F.cross_entropy(
            logits.float(), masked_labels, ignore_index=self.ignore_index, reduction="sum"
        )

    def forward(self, logits: List[torch.Tensor], labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        total_elements = mask.sum()

        # chunk and reshape labels and mask
        labels = labels.chunk(self.num_output_chunks, dim=1)
        mask = mask.chunk(self.num_output_chunks, dim=1)

        # compute one chunk at a time
        total_loss = 0.0
        for logits_chunk, labels_chunk, mask_chunk in zip(logits, labels, mask):
            total_loss += self.compute_cross_entropy(
                logits_chunk.reshape(-1, logits_chunk.size(-1)),
                labels_chunk.reshape(-1),
                mask_chunk.reshape(-1)
            )

        return total_loss / total_elements