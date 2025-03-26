class parameter:
    def __init__(self):
        self.paradic = None

    def para_load(self, zone):
        if zone == 'Single':
            para = {"type": "Single", "Task":"loadpred", "encoLen": 8, "decoLen": 96,
                    "encoder_pHVAC_in": 6, "encoder_pHVAC_h": 20, "encoder_pHVAC_out": 4,
                    "decoder_pHVAC_in": 6, "decoder_pHVAC_h": 20, "decoder_pHVAC_out": 4,
                    "encoder_other_in": 3, "encoder_other_h": 15, "encoder_other_out": 1,
                    "decoder_other_in": 3, "decoder_other_h": 15, "decoder_other_out": 1,
                    "Phvac_in": 4, "Phvac_h1": 8, "Phvac_h2": 8, "Phvac_out": 1,
                    "Ehvac_in": 4, "Ehvac_h1": 8, "Ehvac_h2": 8, "Ehvac_out": 1,
                    "enco_gnn_in": 5, "enco_gnn_h": 15, "enco_gnn_out": 5,
                    "deco_gnn_in": 5, "deco_gnn_h": 15, "deco_gnn_out": 5,
                    }
        if zone == 'Multi':
            para = {"type": "Multi", "Task":"loadpred", "encoLen": 8, "decoLen": 96,
                    "encoder_pHVAC_in": 11, "encoder_pHVAC_h": 20, "encoder_pHVAC_out": 4,
                    "decoder_pHVAC_in": 11, "decoder_pHVAC_h": 20, "decoder_pHVAC_out": 4,
                    "encoder_other_in": 3, "encoder_other_h": 15, "encoder_other_out": 1,
                    "decoder_other_in": 3, "decoder_other_h": 15, "decoder_other_out": 1,
                    "Phvac_in": 4, "Phvac_h1": 8, "Phvac_h2": 8, "Phvac_out": 1,
                    "Ehvac_in": 4, "Ehvac_h1": 8, "Ehvac_h2": 8, "Ehvac_out": 1,
                    "enco_gnn_in": 5, "enco_gnn_h": 15, "enco_gnn_out":5,
                    "deco_gnn_in": 5, "deco_gnn_h": 15, "deco_gnn_out": 5,
                    }
        self.paradic = para
