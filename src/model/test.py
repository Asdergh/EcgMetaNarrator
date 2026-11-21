

if __name__ == "__main__":
    from .backbones import train
    root = "/run/media/ramzan/T7/datasets/ecg/ptb-xl-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    train(model_name="fasternet",
            loader_name="ptbxl",
            root=root,
            epochs=100,
            hf_repo="DeltaRim/corinteltech",
            num_classes=71)