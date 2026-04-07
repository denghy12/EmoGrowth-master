def _is_multi_label_dataset(args):
    dataset = args.get("dataset", "").lower()
    return dataset in {"iscience", "pnas", "neuroimage", "emotic"}


def get_model(model_name, args):
    name = model_name.lower()
    is_ml = _is_multi_label_dataset(args)

    if name == "icarl":
        from models.icarl import iCaRL

        return iCaRL(args)
    elif name == "bic":
        from models.bic import BiC

        return BiC(args)
    elif name == "podnet":
        from models.podnet import PODNet

        return PODNet(args)
    elif name == "lwf":
        if is_ml:
            from models.lwf_ml import LwF as LwF_ML

            return LwF_ML(args)
        from models.lwf import LwF as LwF_SL

        return LwF_SL(args)
    elif name == "ewc":
        if is_ml:
            from models.ewc_ml import EWC as EWC_ML

            return EWC_ML(args)
        from models.ewc import EWC as EWC_SL

        return EWC_SL(args)
    elif name == "wa":
        from models.wa import WA

        return WA(args)
    elif name == "der":
        from models.der import DER

        return DER(args)
    elif name == "finetune":
        if is_ml:
            from models.finetune_ml import Finetune as Finetune_ML

            return Finetune_ML(args)
        from models.finetune import Finetune as Finetune_SL

        return Finetune_SL(args)
    elif name == "replay":
        if is_ml:
            from models.replay_ml import Replay as Replay_ML

            return Replay_ML(args)
        from models.replay import Replay as Replay_SL

        return Replay_SL(args)
    elif name == "gem":
        from models.gem import GEM

        return GEM(args)
    elif name == "coil":
        from models.coil import COIL

        return COIL(args)
    elif name == "foster":
        from models.foster import FOSTER

        return FOSTER(args)
    elif name == "rmm-icarl":
        from models.rmm import RMM_iCaRL

        return RMM_iCaRL(args)
    elif name == "rmm-foster":
        from models.rmm import RMM_FOSTER

        return RMM_FOSTER(args)
    elif name == "fetril":
        from models.fetril import FeTrIL

        return FeTrIL(args)
    elif name == "pass":
        from models.pa2s import PASS

        return PASS(args)
    elif name == "il2a":
        from models.il2a import IL2A

        return IL2A(args)
    elif name == "ssre":
        from models.ssre import SSRE

        return SSRE(args)
    elif name == "memo":
        from models.memo import MEMO

        return MEMO(args)
    elif name == "beefiso":
        from models.beef_iso import BEEFISO

        return BEEFISO(args)
    elif name == "simplecil":
        from models.simplecil import SimpleCIL

        return SimpleCIL(args)
    elif name == "clif":
        from models.clif_ml import CLIF

        return CLIF(args)
    elif name == "agcn":
        from models.agcn_ml import AGCN

        return AGCN(args)
    else:
        raise ValueError(f"Unknown model: {model_name}")
