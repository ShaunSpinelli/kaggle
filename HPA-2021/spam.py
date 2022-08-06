import hpacellseg.cellsegmentator as cellsegmentator


NUC_MODEL = '../input/hpacellsegmentatormodelweights/dpn_unet_nuclei_v1.pth'
CELL_MODEL = '../input/hpacellsegmentatormodelweights/dpn_unet_cell_3ch_v1.pth'

segmentator = cellsegmentator.CellSegmentator(
    NUC_MODEL,
    CELL_MODEL,
    scale_factor=0.25,
    device='cuda',
    padding=True,
    multi_channel_model=True
)


segmentator.pred_nuclei
