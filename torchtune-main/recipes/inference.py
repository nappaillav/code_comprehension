from torchtune.training import FullModelTorchTuneCheckpointer
from torchtune.models.llama3 import llama3_8b

model = llama3_8b()
checkpointer = FullModelTorchTuneCheckpointer(
    checkpoint_dir="/home/vdhee/scratch/LLMcode/Train/Full_Fine_Tuning_Results/output-0.5-0.0",
    checkpoint_files=["hf_model_0001_9.pt", "hf_model_0002_9.pt", "hf_model_0002_9.pt", "hf_model_0002_9.pt"],
    output_dir="/home/vdhee/scratch/LLMcode/Train/Full_Fine_Tuning_Results/output-0.5-0.0/output_dir",
    model_type='LLAMA3',
)
ckpt_dict = checkpointer.load_checkpoint()
model_state_dict = ckpt_dict['model']
model.load_state_dict(model_state_dict)