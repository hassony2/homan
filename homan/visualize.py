import numpy as np
import torch

from homan.utils.nmr_renderer import OrthographicRenderer, PerspectiveRenderer
import neural_renderer as nr


def visualize_perspective(image, predictions, K=None):
    perspect_renderer = PerspectiveRenderer(image_size=max(image.shape))
    new_image = image.copy()
    # 2 * factor to be investigated !
    verts = 2 * torch.Tensor(predictions["verts"]).cuda().unsqueeze(0)
    faces = torch.Tensor(predictions["faces"]).cuda().unsqueeze(0)
    K = torch.Tensor(K).cuda().unsqueeze(0)
    trans = torch.Tensor([0, 0, 0]).cuda().unsqueeze(0)
    for i in range(len(verts)):
        v = verts[i:i + 1]
        new_image = perspect_renderer(vertices=v,
                                      faces=faces,
                                      color_name="blue",
                                      image=new_image,
                                      translation=trans,
                                      K=K)
    return (new_image * 255).astype(np.uint8)


def visualize_orthographic(image, predictions):
    ortho_renderer = OrthographicRenderer(image_size=max(image.shape))
    new_image = image.copy()
    verts = torch.Tensor(predictions["verts"]).cuda().unsqueeze(0)
    faces = torch.Tensor(predictions["faces"]).cuda().unsqueeze(0)
    cams = torch.Tensor(predictions["cams"]).cuda().unsqueeze(0)
    for i in range(len(verts)):
        v = verts[i:i + 1]
        cam = cams[i:i + 1]
        new_image = ortho_renderer(vertices=v,
                                   faces=faces,
                                   cam=cam,
                                   color_name="blue",
                                   image=new_image)
    return (new_image * 255).astype(np.uint8)


def visualize_hand_object(model,
                          images,
                          verts_hand_gt=None,
                          verts_object_gt=None,
                          dist=3,
                          viz_len=7,
                          init=False,
                          gt_only=False,
                          image_size=640,
                          max_in_batch=2):
    if gt_only:
        rends, masks = model.render_gt(
            model.renderer,
            verts_hand_gt=verts_hand_gt,
            verts_object_gt=verts_object_gt,
            viz_len=viz_len,
            max_in_batch=max_in_batch,
        )
    elif verts_hand_gt is None:
        rends, masks = model.render(model.renderer,
                                    viz_len=viz_len,
                                    max_in_batch=max_in_batch)
    else:
        rends, masks = model.render_with_gt(model.renderer,
                                            verts_hand_gt=verts_hand_gt,
                                            verts_object_gt=verts_object_gt,
                                            viz_len=viz_len,
                                            init=init,
                                            max_in_batch=max_in_batch)
    bs = rends.shape[0]
    # Rendered frontal image
    new_images = []
    for image, rend, mask in zip(images, rends, masks):
        if image.max() > 1:
            image = image / 255.0
        h, w, c = image.shape
        L = max(h, w)
        new_image = np.pad(image.copy(), ((0, L - h), (0, L - w), (0, 0)))
        new_image[mask] = rend[mask]
        new_image = (new_image[:h, :w] * 255).astype(np.uint8)
        new_images.append(new_image)

    # Rendered top-down image
    theta = 1.3
    x, y = np.cos(theta), np.sin(theta)
    obj_verts, _ = model.get_verts_object()
    mx, my, mz = obj_verts.mean(dim=(0, 1)).detach().cpu().numpy()

    K = model.renderer.K
    R2 = torch.cuda.FloatTensor([[[1, 0, 0], [0, x, -y], [0, y, x]]])
    t2 = torch.cuda.FloatTensor([mx, my + dist, mz])
    top_renderer = nr.renderer.Renderer(image_size=image_size,
                                        K=K,
                                        R=R2,
                                        t=t2,
                                        orig_size=1)
    top_renderer.background_color = [1, 1, 1]
    top_renderer.light_direction = [1, 0.5, 1]
    top_renderer.light_intensity_direction = 0.3
    top_renderer.light_intensity_ambient = 0.5
    top_renderer.background_color = [1, 1, 1]
    if verts_hand_gt is None:
        top_down, _ = model.render(model.renderer,
                                   rotate=True,
                                   viz_len=viz_len,
                                   max_in_batch=max_in_batch)
    elif gt_only:
        top_down, _ = model.render_gt(
            model.renderer,
            verts_hand_gt=verts_hand_gt,
            verts_object_gt=verts_object_gt,
            viz_len=viz_len,
            rotate=True,
            max_in_batch=max_in_batch,
        )
    else:
        top_down, _ = model.render_with_gt(model.renderer,
                                           verts_hand_gt=verts_hand_gt,
                                           verts_object_gt=verts_object_gt,
                                           rotate=True,
                                           viz_len=viz_len,
                                           init=init,
                                           max_in_batch=max_in_batch)
    top_down = (top_down * 255).astype(np.uint8)
    return np.stack(new_images), top_down
