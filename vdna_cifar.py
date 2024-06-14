from vdna import VDNAProcessor, EMD, load_vdna_from_files
import torch

vdna_proc = VDNAProcessor()
# device = torch.device('cpu')
device = "cpu"
try:
    # vdna1 = vdna_proc.make_vdna(source="C:/09_Master_Program/Uni_Freiburg/Summer_2024/01_DL_lab/Project/Cityscapes/VisualDNA/dataset/cifar10/automobile/data_batch_1_image_4.png", device=device, num_workers=0)
    # vdna2 = vdna_proc.make_vdna(source="C:/09_Master_Program/Uni_Freiburg/Summer_2024/01_DL_lab/Project/Cityscapes/VisualDNA/dataset/cifar10/automobile/data_batch_1_image_4.png", device=device, num_workers=0)
    vdna1 = vdna_proc.make_vdna(source="C:/09_Master_Program/Uni_Freiburg/Summer_2024/01_DL_lab/Project/Cityscapes/VisualDNA/dataset/cifar10/automobile/", device=device, num_workers=0)
    vdna2 = vdna_proc.make_vdna(source="C:/09_Master_Program/Uni_Freiburg/Summer_2024/01_DL_lab/Project/Cityscapes/VisualDNA/dataset/cifar10/airplane/", device=device, num_workers=0)
    vdna1.save("C:/09_Master_Program/Uni_Freiburg/Summer_2024/01_DL_lab/Project/Cityscapes/VisualDNA/dataset/vdna1")
    vdna2.save("C:/09_Master_Program/Uni_Freiburg/Summer_2024/01_DL_lab/Project/Cityscapes/VisualDNA/dataset/vdna2")
except Exception as e:
    print(f"Error: {e}")
# vdna2 = vdna_proc.make_vdna(source="/dataset/cifar10/airplane/data_batch_1_image_4.png", device=device)


# This will load the two files and return a VDNA object

load_vdna1 = load_vdna_from_files("C:/09_Master_Program/Uni_Freiburg/Summer_2024/01_DL_lab/Project/Cityscapes/VisualDNA/dataset/vdna1")
load_vdna2 = load_vdna_from_files("C:/09_Master_Program/Uni_Freiburg/Summer_2024/01_DL_lab/Project/Cityscapes/VisualDNA/dataset/vdna2")


# ----- Checking feature extractor and distribution -----
print(f"vdna1 uses {load_vdna1.feature_extractor_name} as feature extractor and {load_vdna1.name} as distribution.")
print(f"vdna2 uses {load_vdna2.feature_extractor_name} as feature extractor and {load_vdna2.name} as distribution.")


# ----- Checking neurons used -----
print("List of layers and neurons in the VDNA:")
for layer_name in load_vdna1.neurons_list:
	print(f"Layer {layer_name} has {load_vdna1.neurons_list[layer_name]} neurons")
     
for layer_name in load_vdna2.neurons_list:
	print(f"Layer {layer_name} has {load_vdna2.neurons_list[layer_name]} neurons")
      

# ----- Checking distribution values -----
all_neurons_hists = load_vdna1.get_all_neurons_dists()
print(f"We have {all_neurons_hists.shape[0]} neurons in the VDNA1, with {all_neurons_hists.shape[1]} bins each.")
print(f"The highest value in a bin is {all_neurons_hists.max()}")

block_0_neurons_hists = load_vdna1.get_all_neurons_in_layer_dist(layer_name="block_0")
print(f"We have {block_0_neurons_hists.shape[0]} neurons in the VDNA1 using block_0 neurons, with {block_0_neurons_hists.shape[1]} bins each.")
print(f"The highest value in a bin is {block_0_neurons_hists.max()}")

specific_neuron_hist = load_vdna1.get_neuron_dist(layer_name="block_0", neuron_idx=42)
print(f"We have {specific_neuron_hist.shape[1]} bins in the histogram for neuron 42 in block_0.")
print(f"The highest value in a bin is {specific_neuron_hist.max()}")

# ----- Checking distribution values -----
all_neurons_hists = load_vdna2.get_all_neurons_dists()
print(f"We have {all_neurons_hists.shape[0]} neurons in the VDNA2, with {all_neurons_hists.shape[1]} bins each.")
print(f"The highest value in a bin is {all_neurons_hists.max()}")

block_0_neurons_hists = load_vdna2.get_all_neurons_in_layer_dist(layer_name="block_0")
print(f"We have {block_0_neurons_hists.shape[0]} neurons in the VDNA2 using block_0 neurons, with {block_0_neurons_hists.shape[1]} bins each.")
print(f"The highest value in a bin is {block_0_neurons_hists.max()}")

specific_neuron_hist = load_vdna2.get_neuron_dist(layer_name="block_0", neuron_idx=42)
print(f"We have {specific_neuron_hist.shape[1]} bins in the histogram for neuron 42 in block_0.")
print(f"The highest value in a bin is {specific_neuron_hist.max()}")


# ----- Comparing VDNAs -----
# Earth Mover's Distance used to compare histogram-based VDNAs
emd = EMD(load_vdna1, load_vdna2)
print("EMD averaged over all neuron comparisons:", emd)

print("EMD averaged over all neurons of block_0:")
print(EMD(load_vdna1, load_vdna2, use_neurons_from_layer="block_0"))

print("EMD comparing neuron 42 of layer block_0")
print(EMD(load_vdna1, load_vdna2, use_neurons_from_layer="block_0", use_neuron_index=42))


print("Neuron-wise EMD comparisons as a dict:")
emd_neuron_wise = EMD(load_vdna1, load_vdna2, return_neuron_wise=True)
for layer in emd_neuron_wise:
	print(f"EMD using neuron 42 of layer {layer} is {emd_neuron_wise[layer][42]}")