from dataLoader import BlenderDataset


if __name__ == "__main__":
    dataset = BlenderDataset(datadir='/project/jacobcha/nk643/data_src/nerf_synthetic/lego', split='train', downsample=1.0, is_stack=False, N_vis=1)
    n = len(dataset)
    print(n)
    for i in range(750,800):
        sample = dataset[i]
        print("index: ", i)
        print(f"sample rays: {sample['rays'].shape} ")
        print(f"sample rays data {sample['rays']}")
        print(f"sample rgbs: {sample['rgbs'].shape} ")
        print(f"sample rgbs data {sample['rgbs']}")
        if 'masks' in sample:
            print(f"sample masks: {sample['masks'].shape} ")
            print(f"sample masks data {sample['masks']}")
        print()

