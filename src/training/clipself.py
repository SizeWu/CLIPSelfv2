import random
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

class CLIPSelf:
    def __call__(self, batch, model, dist_model, loss, device, cast_dtype, distributed, args):
        if distributed:
            model = model.module
            dist_model = dist_model.module
        images, normed_boxes, image_crops = batch       # note texts are not paired with images

        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        normed_boxes = normed_boxes.to(device=device, dtype=cast_dtype, non_blocking=True)
        image_crops = image_crops.to(device=device, dtype=cast_dtype, non_blocking=True)

        if args.multiscale:
            cur_h, cur_w = images.shape[2:]
            assert cur_h == cur_w
            if cur_h == 1024:
                tar_sizes = [320, 640, 896, 1024]
            elif cur_h == 896:
                tar_sizes = [336, 448, 672, 896]
            else:
                raise NotImplementedError
            tar_size = random.choice(tar_sizes)
            images = F.interpolate(images, size=(tar_size, tar_size), mode='bilinear')

        rois_list = []
        crops_list = []
        for bboxes_per_image, crops_per_image in zip(normed_boxes, image_crops):
            valid = bboxes_per_image[:, -1] > 0.5
            rois_list.append(bboxes_per_image[valid, :4])
            crops_list.append(crops_per_image[valid])

        image_crops = torch.cat(crops_list)
        with torch.no_grad():
            teacher_crop_features = dist_model.encode_image(image_crops, normalize=False)
        student_roi_features = model.encode_pseudo_boxes(images, rois_list, normalize=False,
                                                         extract_type=args.extract_type,
                                                         window_attention=args.window_attention,
                                                         correlative_attention=args.correlative_attention)

        normed_student_features = F.normalize(student_roi_features, dim=-1)
        normed_teacher_features = F.normalize(teacher_crop_features, dim=-1)

        loss_cosine = 1.0 - (normed_student_features *
                             normed_teacher_features).sum(-1).mean()
        losses = dict(loss_cosine=loss_cosine*args.cosine_weight)

        return losses, len(images), model.logit_scale.exp()


class CLIPSelfMask:
    def __call__(self, batch, model, dist_model, loss, device, cast_dtype, distributed, args):
        if distributed:
            model = model.module
            dist_model = dist_model.module
        images, normed_boxes, image_crops, gt_masks = batch       # note texts are not paired with images

        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        normed_boxes = normed_boxes.to(device=device, dtype=cast_dtype, non_blocking=True)
        image_crops = image_crops.to(device=device, dtype=cast_dtype, non_blocking=True)

        rois_list = []
        crops_list = []
        for bboxes_per_image, crops_per_image in zip(normed_boxes, image_crops):
            valid = bboxes_per_image[:, -1] > 0.5
            rois_list.append(bboxes_per_image[valid, :4])
            crops_list.append(crops_per_image[valid])

        image_crops = torch.cat(crops_list)
        with torch.no_grad():
            teacher_crop_features = dist_model.encode_image(image_crops, normalize=True)

        feature_maps = model.encode_dense(images, normalize=True, keep_shape=True,
                                          window_attention=args.window_attention,
                                          correlative_attention=args.correlative_attention)
        student_roi_features = roi_align(feature_maps,
                                         model.visual._denormalize_boxes(normed_boxes, feature_maps),
                                         (1, 1), 1.0, -1, True)[..., 0, 0]

        normed_student_features = F.normalize(student_roi_features, dim=-1)
        normed_teacher_features = F.normalize(teacher_crop_features, dim=-1)

        loss_cosine = 1.0 - (normed_student_features *
                             normed_teacher_features).sum(-1).mean()

        similarities = []
        for gt_masks_per_image, feature_map in zip(gt_masks, feature_maps):
            valid = gt_masks_per_image.sum((-2, -1)) > 2
            feature_map = feature_map.permute(1, 2, 0)
            gt_masks_per_image = gt_masks_per_image[valid]
            for mask in gt_masks_per_image:
                features = feature_map[mask > 0.0]
                similarities.append((features @ features.T).mean())

        loss_smooth = sum(similarities) / len(similarities)

        losses = dict(loss_cosine=loss_cosine*args.cosine_weight,
                      loss_smooth=loss_smooth*args.smooth_weight)

        return losses, len(images), model.logit_scale.exp()
