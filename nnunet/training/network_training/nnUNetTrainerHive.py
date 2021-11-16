#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from collections import OrderedDict
from pathlib import Path
from nnunet.evaluation.evaluator import aggregate_scores
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from batchgenerators.utilities.file_and_folder_operations import *
from Hive.monai.utils.email_utils import send_email
from nnunet.inference.predict import predict_cases
import time
from nnunet.postprocessing.connected_components import determine_postprocessing
from sklearn.model_selection import KFold


class nnUNetTrainerHive(nnUNetTrainerV2):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.summary_writer_tr_loss = None
        self.summary_writer_val_loss = None
        self.summary_writer_eval_metrics = None
        self.save_every = 5
        self.save_prediction_every = 50
        self.config_dict = None

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        self.load_config_dict()
        super().initialize(training, force_load_plans)

        experiment_name = self.config_dict["Experiment Name"]
        self.summary_writer_tr_loss = SummaryWriter(
            log_dir=os.path.join(self.output_folder_base, 'runs', '{}_fold_{}'.format(experiment_name, self.fold)))
        self.summary_writer_val_loss = SummaryWriter(
            log_dir=os.path.join(self.output_folder_base, 'runs', '{}_fold_{}'.format(experiment_name, self.fold)))
        self.summary_writer_eval_metrics = SummaryWriter(
            log_dir=os.path.join(self.output_folder_base, 'runs', '{}_fold_{}'.format(experiment_name, self.fold)))

    def load_config_dict(self):
        json_file = subfiles(os.environ['RESULTS_FOLDER'], suffix='.json')[0]
        print(json_file)
        self.config_dict = load_json(json_file)

    def update_tensorboard_summary(self):
        self.summary_writer_tr_loss.add_scalar("Training Loss", self.all_tr_losses[-1], self.epoch)
        self.summary_writer_tr_loss.flush()
        self.summary_writer_val_loss.add_scalar("Validation Loss", self.all_val_losses[-1], self.epoch)
        self.summary_writer_val_loss.flush()
        self.summary_writer_eval_metrics.add_scalar("Evaluation Metric", self.all_val_eval_metrics[-1], self.epoch)
        self.summary_writer_eval_metrics.flush()

    def run_inference_on_cases(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                               step_size: float = 0.5, use_gaussian: bool = True,
                               all_in_gpu: bool = False,
                               segmentation_export_kwargs: dict = None):
        splits_file = join(self.dataset_directory, "splits_final.pkl")

        if "3D_validation_idx" not in self.config_dict:
            return
        cases_idx = self.config_dict["3D_validation_idx"]

        current_mode = self.network.training
        self.network.eval()

        if segmentation_export_kwargs is None:
            if 'segmentation_export_params' in self.plans.keys():
                force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
                interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
            else:
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0
        else:
            force_separate_z = segmentation_export_kwargs['force_separate_z']
            interpolation_order = segmentation_export_kwargs['interpolation_order']
            interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

        # predictions as they come from the network go here
        output_folder = join(self.output_folder, 'validation_predictions')
        maybe_mkdir_p(output_folder)

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        for idx in cases_idx:
            case = load_pickle(splits_file)[self.fold]['val'][idx]

            properties = load_pickle(os.path.join(self.folder_with_preprocessed_data, case + '.pkl'))
            data = np.load(os.path.join(self.folder_with_preprocessed_data, case + '.npy'))
            data[-1][data[-1] == -1] = 0

            softmax_pred = self.predict_preprocessed_data_return_seg_and_softmax(data[:-1],
                                                                                 do_mirroring=do_mirroring,
                                                                                 mirror_axes=mirror_axes,
                                                                                 use_sliding_window=use_sliding_window,
                                                                                 step_size=step_size,
                                                                                 use_gaussian=use_gaussian,
                                                                                 all_in_gpu=all_in_gpu,
                                                                                 mixed_precision=self.fp16)[1]

            softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])

            softmax_fname = None  # join(output_folder, case + "_{}.npy".format(self.epoch))
            # np.save(join(output_folder, softmax_fname), softmax_pred)
            save_segmentation_nifti_from_softmax(softmax_pred,
                                                 join(output_folder, case + "_{}.nii.gz".format(self.epoch)),
                                                 properties, interpolation_order, self.regions_class_order,
                                                 None, None,
                                                 softmax_fname, None, force_separate_z,
                                                 interpolation_order_z
                                                 )
        self.network.train(current_mode)

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        beginning_time = time.time()
        super().run_training()
        end_time = time.time()

        trainer_state = {"COMPLETED": end_time - beginning_time, "key_metric": "Evaluation_Metric",
                         "val_key_metric_list": self.all_val_eval_metrics,
                         "val_key_metric_alpha": self.val_eval_criterion_alpha, "epoch": self.epoch}
        if "email_password" in os.environ and "email_account" in os.environ and "receiver_email" in os.environ:
            send_email(self.output_folder, os.environ["receiver_email"], self.config_dict["DatasetName"],
                       self.config_dict["Experiment Name"], "", self.fold, trainer_state=trainer_state)

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """
        continue_training = super().on_epoch_end()

        self.update_tensorboard_summary()

        if self.epoch % self.save_prediction_every == (self.save_prediction_every - 1):
            self.run_inference_on_cases()

        return continue_training

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            splits_file = join(self.dataset_directory, "splits_final.pkl")

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file(
                    "Creating new {}-fold cross-validation split...".format(self.config_dict["n_folds"]))
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=self.config_dict["n_folds"], shuffle=True, random_state=self.config_dict["Seed"])
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()
        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True,
                 run_cascade_validation=False):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ret = super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)
        if run_cascade_validation:
            self.print_to_log_file("running cascade validation")
            folder_with_segs_from_prev_stage = Path(self.output_folder_base).parent.parent.parent.joinpath("step_0",
                                                                                                           Path(
                                                                                                               self.output_folder_base).parent.name,
                                                                                                           Path(
                                                                                                               self.output_folder_base).name)
            folder_cascade = Path(self.output_folder_base).parent.parent.parent.joinpath("cascade",
                                                                                         Path(
                                                                                             self.output_folder_base).parent.name,
                                                                                         Path(
                                                                                             self.output_folder_base).name)
            input_list = []
            output_list = []
            pred_gt_tuples = []
            for k in self.dataset_val.keys():
                properties = load_pickle(self.dataset[k]['properties_file'])
                fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
                input_list.append([join(self.config_dict["base_folder"], "nnUNet_raw_data",
                                        "Task{}_{}".format(self.config_dict["Task_ID"], self.config_dict["Task_Name"]),
                                        "imagesTr", "{}_0000.nii.gz".format(fname)),
                                   join(folder_with_segs_from_prev_stage, "fold_{}".format(self.fold),
                                        validation_folder_name + "_postprocessed", fname, fname + ".nii.gz")])
                output_list.append(
                    join(folder_cascade, "fold_{}".format(self.fold), validation_folder_name, fname + ".nii.gz"))
                pred_gt_tuples.append(
                    [str(Path(folder_cascade).joinpath("fold_{}".format(self.fold), validation_folder_name,
                                                       fname + ".nii.gz")),
                     join(self.gt_niftis_folder, fname + ".nii.gz")])

            predict_cases(self.output_folder_base, input_list, output_list, (self.fold,), True, 2, 2,
                          disable_postprocessing=True)

            self.print_to_log_file("finished prediction")

            # evaluate raw predictions
            self.print_to_log_file("evaluation of raw predictions")
            task = self.dataset_directory.split("/")[-1]
            job_name = self.experiment_name

            if not isfile(join(folder_cascade, "fold_{}".format(self.fold), validation_folder_name, "summary.json")):
                _ = aggregate_scores(pred_gt_tuples, labels=list(range(self.num_classes)),
                                     json_output_file=join(folder_cascade, "fold_{}".format(self.fold),
                                                           validation_folder_name, "summary.json"),
                                     json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
                                     json_author="Fabian",
                                     json_task=task, num_threads=5)

            self.print_to_log_file("determining postprocessing")
            determine_postprocessing(join(folder_cascade, "fold_{}".format(self.fold)), self.gt_niftis_folder,
                                     validation_folder_name,
                                     final_subf_name=validation_folder_name + "_postprocessed", debug=debug,
                                     assign_disconnected=True)
