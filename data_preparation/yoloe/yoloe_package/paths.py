INPUT_IMAGES_DIR = 'data/data_sirius/images'
REFS_LOCAL = 'data/tbank_official_logos/refs_ls_coco.json'
OUTPUT_DIR = 'yoloe_results/'
SUBSET = 10  # None for full dataset, or int e.g. 10 for first 10 images (for testing)
RUNS_DIR = 'runs/yoloe_predict'
LABELS_DIR = RUNS_DIR + '/labels'
PSEUDO_COCO = OUTPUT_DIR + 'pseudo_coco.json'