bit_config_dict = {

    "bit_config_resnet18_modelsize_6.7_a6_75B": {
        'quant_input': 8,
        'quant_init_block_convbn': 8,
        'quant_act_int32': 16,

        'stage1.unit1.quant_act': 6,
        'stage1.unit1.quant_convbn1': 8,
        'stage1.unit1.quant_act1': 6,
        'stage1.unit1.quant_convbn2': 8,
        'stage1.unit1.quant_act_int32': 16,

        'stage1.unit2.quant_act': 6,
        'stage1.unit2.quant_convbn1': 8,
        'stage1.unit2.quant_act1': 6,
        'stage1.unit2.quant_convbn2': 8,
        'stage1.unit2.quant_act_int32': 16,

        'stage2.unit1.quant_act': 6,
        'stage2.unit1.quant_convbn1': 7,
        'stage2.unit1.quant_act1': 6,
        'stage2.unit1.quant_convbn2': 8,
        'stage2.unit1.quant_identity_convbn': 8,
        'stage2.unit1.quant_act_int32': 16,

        'stage2.unit2.quant_act': 6,
        'stage2.unit2.quant_convbn1': 8,
        'stage2.unit2.quant_act1': 6,
        'stage2.unit2.quant_convbn2': 8,
        'stage2.unit2.quant_act_int32': 16,

        'stage3.unit1.quant_act': 6,
        'stage3.unit1.quant_convbn1': 8,
        'stage3.unit1.quant_act1': 6,
        'stage3.unit1.quant_convbn2': 4,
        'stage3.unit1.quant_identity_convbn': 4,
        'stage3.unit1.quant_act_int32': 16,

        'stage3.unit2.quant_act': 6,
        'stage3.unit2.quant_convbn1': 6,
        'stage3.unit2.quant_act1': 6,
        'stage3.unit2.quant_convbn2': 8,
        'stage3.unit2.quant_act_int32': 16,

        'stage4.unit1.quant_act': 6,
        'stage4.unit1.quant_convbn1': 4,
        'stage4.unit1.quant_act1': 6,
        'stage4.unit1.quant_convbn2': 4,
        'stage4.unit1.quant_identity_convbn': 4,
        'stage4.unit1.quant_act_int32': 16,

        'stage4.unit2.quant_act': 6,
        'stage4.unit2.quant_convbn1': 4,
        'stage4.unit2.quant_act1': 6,
        'stage4.unit2.quant_convbn2': 4,
        'stage4.unit2.quant_act_int32': 16,

        'quant_act_output': 8,
        'quant_output': 8
    },

    "bit_config_resnet18_modelsize_6.7_a8_97B": {
        'quant_input': 8,
        'quant_init_block_convbn': 8,
        'quant_act_int32': 16,

        'stage1.unit1.quant_act': 8,
        'stage1.unit1.quant_convbn1': 8,
        'stage1.unit1.quant_act1': 8,
        'stage1.unit1.quant_convbn2': 8,
        'stage1.unit1.quant_act_int32': 16,

        'stage1.unit2.quant_act': 8,
        'stage1.unit2.quant_convbn1': 8,
        'stage1.unit2.quant_act1': 8,
        'stage1.unit2.quant_convbn2': 8,
        'stage1.unit2.quant_act_int32': 16,

        'stage2.unit1.quant_act': 8,
        'stage2.unit1.quant_convbn1': 7,
        'stage2.unit1.quant_act1': 8,
        'stage2.unit1.quant_convbn2': 8,
        'stage2.unit1.quant_identity_convbn': 8,
        'stage2.unit1.quant_act_int32': 16,

        'stage2.unit2.quant_act': 8,
        'stage2.unit2.quant_convbn1': 8,
        'stage2.unit2.quant_act1': 8,
        'stage2.unit2.quant_convbn2': 8,
        'stage2.unit2.quant_act_int32': 16,

        'stage3.unit1.quant_act': 8,
        'stage3.unit1.quant_convbn1': 8,
        'stage3.unit1.quant_act1': 8,
        'stage3.unit1.quant_convbn2': 4,
        'stage3.unit1.quant_identity_convbn': 4,
        'stage3.unit1.quant_act_int32': 16,

        'stage3.unit2.quant_act': 8,
        'stage3.unit2.quant_convbn1': 6,
        'stage3.unit2.quant_act1': 8,
        'stage3.unit2.quant_convbn2': 8,
        'stage3.unit2.quant_act_int32': 16,

        'stage4.unit1.quant_act': 8,
        'stage4.unit1.quant_convbn1': 4,
        'stage4.unit1.quant_act1': 8,
        'stage4.unit1.quant_convbn2': 4,
        'stage4.unit1.quant_identity_convbn': 4,
        'stage4.unit1.quant_act_int32': 16,

        'stage4.unit2.quant_act': 8,
        'stage4.unit2.quant_convbn1': 4,
        'stage4.unit2.quant_act1': 8,
        'stage4.unit2.quant_convbn2': 4,
        'stage4.unit2.quant_act_int32': 16,

        'quant_act_output': 8,
        'quant_output': 8
    },


    "bit_config_resnet50_modelsize_16.0_a5_141BOP": {
        'quant_input': 8,
        'quant_init_convbn': 8,
        'quant_act_int32': 16,

        'stage1.unit1.quant_act': 5,
        'stage1.unit1.quant_convbn1': 8,
        'stage1.unit1.quant_act1': 5,
        'stage1.unit1.quant_convbn2': 8,
        'stage1.unit1.quant_act2': 5,
        'stage1.unit1.quant_convbn3': 8,
        'stage1.unit1.quant_identity_convbn': 8,
        'stage1.unit1.quant_act_int32': 16,

        'stage1.unit2.quant_act': 5,
        'stage1.unit2.quant_convbn1': 8,
        'stage1.unit2.quant_act1': 5,
        'stage1.unit2.quant_convbn2': 8,
        'stage1.unit2.quant_act2': 5,
        'stage1.unit2.quant_convbn3': 8,
        'stage1.unit2.quant_act_int32': 16,

        'stage1.unit3.quant_act': 5,
        'stage1.unit3.quant_convbn1': 8,
        'stage1.unit3.quant_act1': 5,
        'stage1.unit3.quant_convbn2': 8,
        'stage1.unit3.quant_act2': 5,
        'stage1.unit3.quant_convbn3': 8,
        'stage1.unit3.quant_act_int32': 16,

        'stage2.unit1.quant_act': 5,
        'stage2.unit1.quant_convbn1': 8,
        'stage2.unit1.quant_act1': 5,
        'stage2.unit1.quant_convbn2': 8,
        'stage2.unit1.quant_act2': 5,
        'stage2.unit1.quant_convbn3': 8,
        'stage2.unit1.quant_identity_convbn': 8,
        'stage2.unit1.quant_act_int32': 16,

        'stage2.unit2.quant_act': 5,
        'stage2.unit2.quant_convbn1': 8,
        'stage2.unit2.quant_act1': 5,
        'stage2.unit2.quant_convbn2': 8,
        'stage2.unit2.quant_act2': 5,
        'stage2.unit2.quant_convbn3': 8,
        'stage2.unit2.quant_act_int32': 16,

        'stage2.unit3.quant_act': 5,
        'stage2.unit3.quant_convbn1': 8,
        'stage2.unit3.quant_act1': 5,
        'stage2.unit3.quant_convbn2': 8,
        'stage2.unit3.quant_act2': 5,
        'stage2.unit3.quant_convbn3': 8,
        'stage2.unit3.quant_act_int32': 16,

        'stage2.unit4.quant_act': 5,
        'stage2.unit4.quant_convbn1': 8,
        'stage2.unit4.quant_act1': 5,
        'stage2.unit4.quant_convbn2': 8,
        'stage2.unit4.quant_act2': 5,
        'stage2.unit4.quant_convbn3': 8,
        'stage2.unit4.quant_act_int32': 16,

        'stage3.unit1.quant_act': 5,
        'stage3.unit1.quant_convbn1': 8,
        'stage3.unit1.quant_act1': 5,
        'stage3.unit1.quant_convbn2': 4,
        'stage3.unit1.quant_act2': 5,
        'stage3.unit1.quant_convbn3': 4,
        'stage3.unit1.quant_identity_convbn': 4,
        'stage3.unit1.quant_act_int32': 16,

        'stage3.unit2.quant_act': 5,
        'stage3.unit2.quant_convbn1': 8,
        'stage3.unit2.quant_act1': 5,
        'stage3.unit2.quant_convbn2': 4,
        'stage3.unit2.quant_act2': 5,
        'stage3.unit2.quant_convbn3': 8,
        'stage3.unit2.quant_act_int32': 16,

        'stage3.unit3.quant_act': 5,
        'stage3.unit3.quant_convbn1': 8,
        'stage3.unit3.quant_act1': 5,
        'stage3.unit3.quant_convbn2': 4,
        'stage3.unit3.quant_act2': 5,
        'stage3.unit3.quant_convbn3': 8,
        'stage3.unit3.quant_act_int32': 16,

        'stage3.unit4.quant_act': 5,
        'stage3.unit4.quant_convbn1': 8,
        'stage3.unit4.quant_act1': 5,
        'stage3.unit4.quant_convbn2': 5,
        'stage3.unit4.quant_act2': 5,
        'stage3.unit4.quant_convbn3': 8,
        'stage3.unit4.quant_act_int32': 16,

        'stage3.unit5.quant_act': 5,
        'stage3.unit5.quant_convbn1': 8,
        'stage3.unit5.quant_act1': 5,
        'stage3.unit5.quant_convbn2': 8,
        'stage3.unit5.quant_act2': 5,
        'stage3.unit5.quant_convbn3': 8,
        'stage3.unit5.quant_act_int32': 16,

        'stage3.unit6.quant_act': 5,
        'stage3.unit6.quant_convbn1': 8,
        'stage3.unit6.quant_act1': 5,
        'stage3.unit6.quant_convbn2': 4,
        'stage3.unit6.quant_act2': 5,
        'stage3.unit6.quant_convbn3': 8,
        'stage3.unit6.quant_act_int32': 16,

        'stage4.unit1.quant_act': 5,
        'stage4.unit1.quant_convbn1': 4,
        'stage4.unit1.quant_act1': 5,
        'stage4.unit1.quant_convbn2': 4,
        'stage4.unit1.quant_act2': 5,
        'stage4.unit1.quant_convbn3': 4,
        'stage4.unit1.quant_identity_convbn': 4,
        'stage4.unit1.quant_act_int32': 16,

        'stage4.unit2.quant_act': 5,
        'stage4.unit2.quant_convbn1': 4,
        'stage4.unit2.quant_act1': 5,
        'stage4.unit2.quant_convbn2': 4,
        'stage4.unit2.quant_act2': 5,
        'stage4.unit2.quant_convbn3': 4,
        'stage4.unit2.quant_act_int32': 16,

        'stage4.unit3.quant_act': 5,
        'stage4.unit3.quant_convbn1': 4,
        'stage4.unit3.quant_act1': 5,
        'stage4.unit3.quant_convbn2': 4,
        'stage4.unit3.quant_act2': 5,
        'stage4.unit3.quant_convbn3': 8,
        'stage4.unit3.quant_act_int32': 16,

        'quant_act_output': 8,
        'quant_output': 8
    },

    "bit_config_resnet50_modelsize_18.7_a5_156BOP": {
        'quant_input': 8,
        'quant_init_convbn': 8,
        'quant_act_int32': 16,

        'stage1.unit1.quant_act': 5,
        'stage1.unit1.quant_convbn1': 8,
        'stage1.unit1.quant_act1': 5,
        'stage1.unit1.quant_convbn2': 8,
        'stage1.unit1.quant_act2': 5,
        'stage1.unit1.quant_convbn3': 8,
        'stage1.unit1.quant_identity_convbn': 8,
        'stage1.unit1.quant_act_int32': 16,

        'stage1.unit2.quant_act': 5,
        'stage1.unit2.quant_convbn1': 8,
        'stage1.unit2.quant_act1': 5,
        'stage1.unit2.quant_convbn2': 8,
        'stage1.unit2.quant_act2': 5,
        'stage1.unit2.quant_convbn3': 8,
        'stage1.unit2.quant_act_int32': 16,

        'stage1.unit3.quant_act': 5,
        'stage1.unit3.quant_convbn1': 8,
        'stage1.unit3.quant_act1': 5,
        'stage1.unit3.quant_convbn2': 8,
        'stage1.unit3.quant_act2': 5,
        'stage1.unit3.quant_convbn3': 8,
        'stage1.unit3.quant_act_int32': 16,

        'stage2.unit1.quant_act': 5,
        'stage2.unit1.quant_convbn1': 8,
        'stage2.unit1.quant_act1': 5,
        'stage2.unit1.quant_convbn2': 8,
        'stage2.unit1.quant_act2': 5,
        'stage2.unit1.quant_convbn3': 8,
        'stage2.unit1.quant_identity_convbn': 8,
        'stage2.unit1.quant_act_int32': 16,

        'stage2.unit2.quant_act': 5,
        'stage2.unit2.quant_convbn1': 8,
        'stage2.unit2.quant_act1': 5,
        'stage2.unit2.quant_convbn2': 8,
        'stage2.unit2.quant_act2': 5,
        'stage2.unit2.quant_convbn3': 8,
        'stage2.unit2.quant_act_int32': 16,

        'stage2.unit3.quant_act': 5,
        'stage2.unit3.quant_convbn1': 8,
        'stage2.unit3.quant_act1': 5,
        'stage2.unit3.quant_convbn2': 8,
        'stage2.unit3.quant_act2': 5,
        'stage2.unit3.quant_convbn3': 8,
        'stage2.unit3.quant_act_int32': 16,

        'stage2.unit4.quant_act': 5,
        'stage2.unit4.quant_convbn1': 8,
        'stage2.unit4.quant_act1': 5,
        'stage2.unit4.quant_convbn2': 8,
        'stage2.unit4.quant_act2': 5,
        'stage2.unit4.quant_convbn3': 8,
        'stage2.unit4.quant_act_int32': 16,

        'stage3.unit1.quant_act': 5,
        'stage3.unit1.quant_convbn1': 8,
        'stage3.unit1.quant_act1': 5,
        'stage3.unit1.quant_convbn2': 8,
        'stage3.unit1.quant_act2': 5,
        'stage3.unit1.quant_convbn3': 4,
        'stage3.unit1.quant_identity_convbn': 4,
        'stage3.unit1.quant_act_int32': 16,

        'stage3.unit2.quant_act': 5,
        'stage3.unit2.quant_convbn1': 8,
        'stage3.unit2.quant_act1': 5,
        'stage3.unit2.quant_convbn2': 8,
        'stage3.unit2.quant_act2': 5,
        'stage3.unit2.quant_convbn3': 8,
        'stage3.unit2.quant_act_int32': 16,

        'stage3.unit3.quant_act': 5,
        'stage3.unit3.quant_convbn1': 8,
        'stage3.unit3.quant_act1': 5,
        'stage3.unit3.quant_convbn2': 8,
        'stage3.unit3.quant_act2': 5,
        'stage3.unit3.quant_convbn3': 8,
        'stage3.unit3.quant_act_int32': 16,

        'stage3.unit4.quant_act': 5,
        'stage3.unit4.quant_convbn1': 8,
        'stage3.unit4.quant_act1': 5,
        'stage3.unit4.quant_convbn2': 8,
        'stage3.unit4.quant_act2': 5,
        'stage3.unit4.quant_convbn3': 8,
        'stage3.unit4.quant_act_int32': 16,

        'stage3.unit5.quant_act': 5,
        'stage3.unit5.quant_convbn1': 8,
        'stage3.unit5.quant_act1': 5,
        'stage3.unit5.quant_convbn2': 8,
        'stage3.unit5.quant_act2': 5,
        'stage3.unit5.quant_convbn3': 8,
        'stage3.unit5.quant_act_int32': 16,

        'stage3.unit6.quant_act': 5,
        'stage3.unit6.quant_convbn1': 8,
        'stage3.unit6.quant_act1': 5,
        'stage3.unit6.quant_convbn2': 8,
        'stage3.unit6.quant_act2': 5,
        'stage3.unit6.quant_convbn3': 8,
        'stage3.unit6.quant_act_int32': 16,

        'stage4.unit1.quant_act': 5,
        'stage4.unit1.quant_convbn1': 8,
        'stage4.unit1.quant_act1': 5,
        'stage4.unit1.quant_convbn2': 4,
        'stage4.unit1.quant_act2': 5,
        'stage4.unit1.quant_convbn3': 4,
        'stage4.unit1.quant_identity_convbn': 4,
        'stage4.unit1.quant_act_int32': 16,

        'stage4.unit2.quant_act': 5,
        'stage4.unit2.quant_convbn1': 5,
        'stage4.unit2.quant_act1': 5,
        'stage4.unit2.quant_convbn2': 4,
        'stage4.unit2.quant_act2': 5,
        'stage4.unit2.quant_convbn3': 8,
        'stage4.unit2.quant_act_int32': 16,

        'stage4.unit3.quant_act': 5,
        'stage4.unit3.quant_convbn1': 8,
        'stage4.unit3.quant_act1': 5,
        'stage4.unit3.quant_convbn2': 4,
        'stage4.unit3.quant_act2': 5,
        'stage4.unit3.quant_convbn3': 8,
        'stage4.unit3.quant_act_int32': 16,

        'quant_act_output': 8,
        'quant_output': 8
    },

    "bit_config_resnet50_modelsize_21.2_a7_226BOP": {
        'quant_input': 8,
        'quant_init_convbn': 8,
        'quant_act_int32': 16,

        'stage1.unit1.quant_act': 7,
        'stage1.unit1.quant_convbn1': 8,
        'stage1.unit1.quant_act1': 7,
        'stage1.unit1.quant_convbn2': 8,
        'stage1.unit1.quant_act2': 7,
        'stage1.unit1.quant_convbn3': 8,
        'stage1.unit1.quant_identity_convbn': 8,
        'stage1.unit1.quant_act_int32': 16,

        'stage1.unit2.quant_act': 7,
        'stage1.unit2.quant_convbn1': 8,
        'stage1.unit2.quant_act1': 7,
        'stage1.unit2.quant_convbn2': 8,
        'stage1.unit2.quant_act2': 7,
        'stage1.unit2.quant_convbn3': 8,
        'stage1.unit2.quant_act_int32': 16,

        'stage1.unit3.quant_act': 7,
        'stage1.unit3.quant_convbn1': 8,
        'stage1.unit3.quant_act1': 7,
        'stage1.unit3.quant_convbn2': 8,
        'stage1.unit3.quant_act2': 7,
        'stage1.unit3.quant_convbn3': 8,
        'stage1.unit3.quant_act_int32': 16,

        'stage2.unit1.quant_act': 7,
        'stage2.unit1.quant_convbn1': 8,
        'stage2.unit1.quant_act1': 7,
        'stage2.unit1.quant_convbn2': 8,
        'stage2.unit1.quant_act2': 7,
        'stage2.unit1.quant_convbn3': 8,
        'stage2.unit1.quant_identity_convbn': 8,
        'stage2.unit1.quant_act_int32': 16,

        'stage2.unit2.quant_act': 7,
        'stage2.unit2.quant_convbn1': 8,
        'stage2.unit2.quant_act1': 7,
        'stage2.unit2.quant_convbn2': 8,
        'stage2.unit2.quant_act2': 7,
        'stage2.unit2.quant_convbn3': 8,
        'stage2.unit2.quant_act_int32': 16,

        'stage2.unit3.quant_act': 7,
        'stage2.unit3.quant_convbn1': 8,
        'stage2.unit3.quant_act1': 7,
        'stage2.unit3.quant_convbn2': 8,
        'stage2.unit3.quant_act2': 7,
        'stage2.unit3.quant_convbn3': 8,
        'stage2.unit3.quant_act_int32': 16,

        'stage2.unit4.quant_act': 7,
        'stage2.unit4.quant_convbn1': 8,
        'stage2.unit4.quant_act1': 7,
        'stage2.unit4.quant_convbn2': 8,
        'stage2.unit4.quant_act2': 7,
        'stage2.unit4.quant_convbn3': 8,
        'stage2.unit4.quant_act_int32': 16,

        'stage3.unit1.quant_act': 7,
        'stage3.unit1.quant_convbn1': 8,
        'stage3.unit1.quant_act1': 7,
        'stage3.unit1.quant_convbn2': 8,
        'stage3.unit1.quant_act2': 7,
        'stage3.unit1.quant_convbn3': 8,
        'stage3.unit1.quant_identity_convbn': 8,
        'stage3.unit1.quant_act_int32': 16,

        'stage3.unit2.quant_act': 7,
        'stage3.unit2.quant_convbn1': 8,
        'stage3.unit2.quant_act1': 7,
        'stage3.unit2.quant_convbn2': 8,
        'stage3.unit2.quant_act2': 7,
        'stage3.unit2.quant_convbn3': 8,
        'stage3.unit2.quant_act_int32': 16,

        'stage3.unit3.quant_act': 7,
        'stage3.unit3.quant_convbn1': 8,
        'stage3.unit3.quant_act1': 7,
        'stage3.unit3.quant_convbn2': 8,
        'stage3.unit3.quant_act2': 7,
        'stage3.unit3.quant_convbn3': 8,
        'stage3.unit3.quant_act_int32': 16,

        'stage3.unit4.quant_act': 7,
        'stage3.unit4.quant_convbn1': 8,
        'stage3.unit4.quant_act1': 7,
        'stage3.unit4.quant_convbn2': 8,
        'stage3.unit4.quant_act2': 7,
        'stage3.unit4.quant_convbn3': 8,
        'stage3.unit4.quant_act_int32': 16,

        'stage3.unit5.quant_act': 7,
        'stage3.unit5.quant_convbn1': 8,
        'stage3.unit5.quant_act1': 7,
        'stage3.unit5.quant_convbn2': 8,
        'stage3.unit5.quant_act2': 7,
        'stage3.unit5.quant_convbn3': 8,
        'stage3.unit5.quant_act_int32': 16,

        'stage3.unit6.quant_act': 7,
        'stage3.unit6.quant_convbn1': 8,
        'stage3.unit6.quant_act1': 7,
        'stage3.unit6.quant_convbn2': 8,
        'stage3.unit6.quant_act2': 7,
        'stage3.unit6.quant_convbn3': 8,
        'stage3.unit6.quant_act_int32': 16,

        'stage4.unit1.quant_act': 7,
        'stage4.unit1.quant_convbn1': 8,
        'stage4.unit1.quant_act1': 7,
        'stage4.unit1.quant_convbn2': 4,
        'stage4.unit1.quant_act2': 7,
        'stage4.unit1.quant_convbn3': 4,
        'stage4.unit1.quant_identity_convbn': 4,
        'stage4.unit1.quant_act_int32': 16,

        'stage4.unit2.quant_act': 7,
        'stage4.unit2.quant_convbn1': 8,
        'stage4.unit2.quant_act1': 7,
        'stage4.unit2.quant_convbn2': 6,
        'stage4.unit2.quant_act2': 7,
        'stage4.unit2.quant_convbn3': 8,
        'stage4.unit2.quant_act_int32': 16,

        'stage4.unit3.quant_act': 7,
        'stage4.unit3.quant_convbn1': 8,
        'stage4.unit3.quant_act1': 7,
        'stage4.unit3.quant_convbn2': 8,
        'stage4.unit3.quant_act2': 7,
        'stage4.unit3.quant_convbn3': 8,
        'stage4.unit3.quant_act_int32': 16,

        'quant_act_output': 8,
        'quant_output': 8
    },

}
