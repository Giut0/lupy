from torchvision import transforms

CONFIDENCE_THRESHOLD_EARLY_STOP = 0.80

label_map = {
    'badger': 0, 'bird': 1, 'boar': 2, 'butterfly': 3,
    'cat': 4, 'dog': 5, 'fox': 6, 'lizard': 7,
    'podolic_cow': 8, 'porcupine': 9, 'weasel': 10,
    'wolf': 11, 'other': 12
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
