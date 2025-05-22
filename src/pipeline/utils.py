import random
import time
    
def generate_model_name(
    project="AMRPred",
    model_type="DNN",
    encoding="KMER",
    model_unique_id=random.randint(1000, 9999),
    epochs=20,
    tag="EXPERIMENTAL",
    index=None,
    run_id=time.time()
):
    """
    Generate a standardized model name for saving or tracking purposes.
    
    Example: AMRPred-DNN-KMER-v0.1-20ep-EXPERIMENTAL-20250521
    """
    parts = [
        project,
        model_type,
        encoding,
        str(model_unique_id),
        f"{epochs}ep",
        tag.upper(),
        run_id
    ]

    return "-".join(parts)





