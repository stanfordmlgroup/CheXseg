# constants for localization evaluation
group_dir = '/deep/group/aihc-bootcamp-spring2020/localize'
valid_cam_dirs = {'gradcam_single':'/deep/group/aihc-bootcamp-spring2020/localize/densenet_single/best_densenet_single_ckpt_epoch=0-chexpert_competition_AUROC=0.88.ckpt_valid/cams/',
                     'gradcam_ensemble': '/deep/group/aihc-bootcamp-spring2020/localize/uncertainty_handling/valid_predictions/ensemble_results/cams/',
                     'ig_ensemble': f'{group_dir}/ig_results/ig_ensemble_valid/cams/'}
    
test_cam_dirs = {'gradcam_single':f'{group_dir}/densenet_single/best_densenet_single_ckpt_epoch=0-chexpert_competition_AUROC=0.88.ckpt_test/cams/',
                     'gradcam_ensemble':f'{group_dir}/uncertainty_handling/test_predictions/ensemble_results/cams/',
                     'ig_ensemble': f'{group_dir}/ig_results/ig_ensemble_test/cams/',
                     'ignt_ensemble': f'{group_dir}/ig_results/ignt_ensemble_test/cams/',
                     'gradcamnt_ensemble': f'{group_dir}/gradcam_nt/cams/'}

LOCALIZATION_TASKS =  ["Enlarged Cardiomediastinum",
                  "Cardiomegaly",
                  "Lung Lesion",
                  "Airspace Opacity",
                  "Edema",
                  "Consolidation",
                  "Atelectasis",
                  "Pneumothorax",
                  "Pleural Effusion",
                  "Support Devices"
                  ]

CHEXPERT_TASKS = ["No Finding",
                  "Enlarged Cardiomediastinum",
                  "Cardiomegaly",
                  "Lung Lesion",
                  "Airspace Opacity",
                  "Edema",
                  "Consolidation",
                  "Pneumonia",
                  "Atelectasis",
                  "Pneumothorax",
                  "Pleural Effusion",
                  "Pleural Other",
                  "Fracture",
                  "Support Devices"
                  ]

CHEXPERT_COMPETITION_TASKS = ["Atelectasis",
                              "Cardiomegaly",
                              "Consolidation",
                              "Edema",
                              "Pleural Effusion"
                              ]

IG_CUTOFF = {'Enlarged Cardiomediastinum': 0.0003034546925302229,
 'Cardiomegaly': 0.0010329889030092291,
 'Lung Lesion': 0.005012613593831528,
 'Airspace Opacity': 0.001378735858732118,
 'Edema': 0.0025567774159403955,
 'Consolidation': 0.0010251582074814883,
 'Atelectasis': 0.0014392538473774219,
 'Pneumothorax': 0.009772481571914802,
 'Pleural Effusion': 0.004837835450362191,
 'Support Devices': 0.0029983863654411123}

IG_CUTOFF_2 = {'Enlarged Cardiomediastinum': 0.00023552914810243554,
 'Cardiomegaly': 0.0007338481259723314,
 'Lung Lesion': 0.000993995173858145,
 'Airspace Opacity': 0.001139621338986147,
 'Edema': 0.0020307370318243135,
 'Consolidation': 0.0007304732822096702,
 'Atelectasis': 0.0012160822276511573,
 'Pneumothorax': 0.0037899313556253285,
 'Pleural Effusion': 0.0033582639621777665,
 'Support Devices': 0.0024224646998426093}

IG_CUTOFF_3 = {'Enlarged Cardiomediastinum': 0.00019357956171784009,
 'Cardiomegaly': 0.0005079578766479897,
 'Lung Lesion': 0.0005894249052895854,
 'Airspace Opacity': 0.0009948079904111532,
 'Edema': 0.001493285191603578,
 'Consolidation': 0.0005253875359463005,
 'Atelectasis': 0.0010147681380184288,
 'Pneumothorax': 0.0018101672962323814,
 'Pleural Effusion': 0.002588135196875112,
 'Support Devices': 0.0019131405254165267}

IG_NT_CUTOFF = {'Enlarged Cardiomediastinum': 2.976040134768465e-07,
 'Cardiomegaly': 2.167249820745499e-06,
 'Lung Lesion': 3.5144783489671776e-06,
 'Airspace Opacity': 4.668738487313161e-06,
 'Edema': 1.7648466256594895e-05,
 'Consolidation': 1.087256324493725e-06,
 'Atelectasis': 3.824014717023462e-06,
 'Pneumothorax': 4.464757640585631e-06,
 'Pleural Effusion': 2.404192846758951e-05,
 'Support Devices': 3.350040103377513e-05}

GRADCAM_NT_CUTOFF = {'Enlarged Cardiomediastinum': 9.268495699327811e-09,
 'Cardiomegaly': 3.585516416800267e-08,
 'Lung Lesion': 2.8652201182963777e-08,
 'Airspace Opacity': 1.0101292209933147e-06,
 'Edema': 1.3907804993952872e-05,
 'Consolidation': 2.4323141681966615e-08,
 'Atelectasis': 5.868654036534373e-07,
 'Pneumothorax': 4.2463043078336903e-07,
 'Pleural Effusion': 3.8713034632953487e-07,
 'Support Devices': 2.0588438148564838e-07}

GRADCAM_CUTOFF = {'Enlarged Cardiomediastinum': 0.00017738173816265522,
 'Cardiomegaly': 0.00042322561098561166,
 'Lung Lesion': 0.0008633324958017875,
 'Airspace Opacity': 0.0008453797374386341,
 'Edema': 0.0013324563642247364,
 'Consolidation': 0.00032908820968049655,
 'Atelectasis': 0.0008193205324654331,
 'Pneumothorax': 0.0013976036161025128,
 'Pleural Effusion': 0.0011708297424464144,
 'Support Devices': 0.0011636815463694243}
