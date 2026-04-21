def get_model(model_name, args):
    name = model_name.lower()
    if name == "icarl":
        from models.icarl import iCaRL
        return iCaRL(args)
    elif name == "podnet":
        from models.podnet import PODNet
        return PODNet(args)
    elif name == "lwf":
        from models.lwf import LwF
        return LwF(args)
    elif name == "ewc":
        from models.ewc import EWC
        return EWC(args)
    elif name == "der":
        from models.der import DER
        return DER(args)
    elif name == "finetune":
        from models.finetune import Finetune
        return Finetune(args)
    elif name == "replay":
        from models.replay import Replay
        return Replay(args)
    elif name == "gem":
        from models.gem import GEM
        return GEM(args)
    elif name == "rmm-icarl":
        from models.rmm import RMM_FOSTER, RMM_iCaRL
        return RMM_iCaRL(args)
    elif name == "rmm-foster":
        from models.rmm import RMM_FOSTER, RMM_iCaRL
        return RMM_FOSTER(args)
    elif name == "tagfex":
        from models.tagfex import TagFex
        return TagFex(args)
    elif name == "tagfex_lejepa":
        from models.tagfex_lejepa import TagFex
        return TagFex(args)
    elif name == "tagfex_der":
        from models.tagfex_der import TagFex
        return TagFex(args)
    elif name == "tagfex_ta_cls":
        from models.tagfex_ta_cls import TagFexTACls
        return TagFexTACls(args)
    elif name == "tagfex_ta_detach":
        from models.tagfex_ta_detach import TagFexTADetach
        return TagFexTADetach(args)
    elif name == "tagfex_ta_detach_lejepa":
        from models.tagfex_ta_detach_lejepa import TagFexTADetachLeJEPA
        return TagFexTADetachLeJEPA(args)
    elif name == "tagfex_simplified_mean_wo_transfer":
        from models.tagfex_simplified_mean_wo_transfer import TagFexSimplifiedMeanWoTransfer
        return TagFexSimplifiedMeanWoTransfer(args)
    elif name == "tagfex_replace_ta_with_task0_ts":
        from models.tagfex_replace_ta_with_task0_ts import TagFexReplaceTAWithTask0TS
        return TagFexReplaceTAWithTask0TS(args)
    elif name == "tagfex_simplified_replace_ta_with_task0_ts":
        from models.tagfex_simplified_replace_ta_with_task0_ts import TagFexSimplifiedReplaceTAWithTask0TS
        return TagFexSimplifiedReplaceTAWithTask0TS(args)
    else:
        assert 0
