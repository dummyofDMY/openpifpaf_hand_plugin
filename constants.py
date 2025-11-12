import os

import numpy as np
try:
    import matplotlib.cm as mplcm
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    pass

import openpifpaf

from .transforms import transform_skeleton

HAND_KEYPOINTS = [
    "wrist_0",
    "wrist_1",
    "thump_mcp",
    "thump_pip",
    "thump_dip",
    "index_mcp_0",
    "index_mcp_1",
    "index_pip",
    "index_dip",
    "middle_mcp_0",
    "middle_mcp_1",
    "middle_pip",
    "middle_dip",
    "ring_mcp_0",
    "ring_mcp_1",
    "ring_pip",
    "ring_dip",
    "pinky_mcp_0",
    "pinky_mcp_1",
    "pinky_pip",
    "pinky_dip"
]

HAND_SKELETON = [
    (0, 1), (1, 2), (2, 3),
    (3, 4), (2, 5), (5, 6),
    (6, 7), (7, 8), (5, 9),
    (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18),
    (18, 19), (19, 20), (1, 17)
]
# 额外的密集骨架
HAND_SKELETON += [
    (1, 9), (1, 13),
    (3, 7), (7, 11), (11, 15), (15, 19),
    (4, 8), (8, 12), (12, 16), (16, 20),
    (6, 10), (10, 14), (14, 18),
    (1, 4), (1, 8), (1, 12), (1, 16), (1, 20)
]
HAND_SKELETON = [(bone[0] + 1, bone[1] + 1) for bone in HAND_SKELETON]  # WHAT CAN I SAY

HAND_SIGMAS = [0.05] * len(HAND_KEYPOINTS)

HAND_SCORE_WEIGHTS = [1.0] * len(HAND_KEYPOINTS)
assert len(HAND_SCORE_WEIGHTS) == len(HAND_KEYPOINTS)

HAND_HFLIP = {}

HAND_CATEGORIES = ['hand']

# x = [-3, 3] y = [0,4]
HAND_POSE = np.array([
    [0, 0, 0],  # wrist_0
    [0, 1, 0],  # wrist_1
    [1.5, 2, 0],  # thumb_mcp
    [2.5, 2, 0],  # thumb_pip
    [3.5, 3, 0],  # thumb_dip
    [1.5, 3, 0],  # index_mcp_0
    [1.5, 4, 0],  # index_mcp_1
    [1.5, 5, 0],  # index_pip
    [1.5, 6, 0],  # index_dip
    [0.5, 3, 0],  # middle_mcp_0
    [0.5, 4, 0],  # middle_mcp_1
    [0.5, 5, 0],  # middle_pip
    [0.5, 6, 0],  # middle_dip
    [-0.5, 3, 0],  # ring_mcp_0
    [-0.5, 4, 0],  # ring_mcp_1
    [-0.5, 5, 0],  # ring_pip
    [-0.5, 6, 0],  # ring_dip
    [-1.5, 3, 0],  # pinky_mcp_0
    [-1.5, 4, 0],  # pinky_mcp_1
    [-1.5, 5, 0],  # pinky_pip
    [-1.5, 6, 0],  # pinky_dip
])

def get_constants():
    HAND_POSE[:, 2] = 0.5
    return [HAND_KEYPOINTS, HAND_SKELETON, HAND_HFLIP, HAND_SIGMAS,
                HAND_POSE, HAND_CATEGORIES, HAND_SCORE_WEIGHTS]


def draw_ann(ann, *, keypoint_painter, filename=None, margin=0.5, aspect=None, **kwargs):
    from openpifpaf import show  # pylint: disable=import-outside-toplevel

    bbox = ann.bbox()
    xlim = bbox[0] - margin, bbox[0] + bbox[2] + margin
    ylim = bbox[1] - margin, bbox[1] + bbox[3] + margin
    if aspect == 'equal':
        fig_w = 5.0
    else:
        fig_w = 5.0 / (ylim[1] - ylim[0]) * (xlim[1] - xlim[0])

    with show.canvas(filename, figsize=(fig_w, 5), nomargin=True, **kwargs) as ax:
        ax.set_axis_off()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        if aspect is not None:
            ax.set_aspect(aspect)

        keypoint_painter.annotation(ax, ann)


def draw_skeletons(pose, sigmas, skel, kps, scr_weights):
    from openpifpaf.annotation import Annotation  # pylint: disable=import-outside-toplevel
    from openpifpaf import show  # pylint: disable=import-outside-toplevel

    scale = np.sqrt(
        (np.max(pose[:, 0]) - np.min(pose[:, 0]))
        * (np.max(pose[:, 1]) - np.min(pose[:, 1]))
    )

    show.KeypointPainter.show_joint_scales = True
    keypoint_painter = show.KeypointPainter()
    ann = Annotation(keypoints=kps, skeleton=skel, score_weights=scr_weights)
    ann.set(pose, np.array(sigmas) * scale)
    os.makedirs('docs', exist_ok=True)
    draw_ann(ann, filename='docs/skeleton_car.png', keypoint_painter=keypoint_painter)


def plot3d_red(ax_2D, p3d, skeleton):
    skeleton = [(bone[0] - 1, bone[1] - 1) for bone in skeleton]

    rot_p90_x = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    p3d = p3d @ rot_p90_x

    fig = ax_2D.get_figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.set_axis_off()
    ax_2D.set_axis_off()

    ax.view_init(azim=-90, elev=20)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    max_range = np.array([p3d[:, 0].max() - p3d[:, 0].min(),
                          p3d[:, 1].max() - p3d[:, 1].min(),
                          p3d[:, 2].max() - p3d[:, 2].min()]).max() / 2.0
    mid_x = (p3d[:, 0].max() + p3d[:, 0].min()) * 0.5
    mid_y = (p3d[:, 1].max() + p3d[:, 1].min()) * 0.5
    mid_z = (p3d[:, 2].max() + p3d[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)  # pylint: disable=no-member

    for ci, bone in enumerate(skeleton):
        c = mplcm.get_cmap('tab20')((ci % 20 + 0.05) / 20)  # Same coloring as Pifpaf preds
        ax.plot(p3d[bone, 0], p3d[bone, 1], p3d[bone, 2], color=c)

    def animate(i):
        ax.view_init(elev=10., azim=i)
        return fig

    return FuncAnimation(fig, animate, frames=360, interval=100)


def print_associations():
    print("\nAssociations of the hand skeleton with 21 keypoints")
    for j1, j2 in HAND_SKELETON:
        print(HAND_KEYPOINTS[j1 - 1], '-', HAND_KEYPOINTS[j2 - 1])


def main():
    print_associations()
# =============================================================================
#     draw_skeletons(CAR_POSE_24, sigmas = CAR_SIGMAS_24, skel = CAR_SKELETON_24,
#                    kps = CAR_KEYPOINTS_24, scr_weights = CAR_SCORE_WEIGHTS_24)
#     draw_skeletons(CAR_POSE_66, sigmas = CAR_SIGMAS_66, skel = CAR_SKELETON_66,
#                    kps = CAR_KEYPOINTS_66, scr_weights = CAR_SCORE_WEIGHTS_66)
# =============================================================================
    with openpifpaf.show.Canvas.blank(nomargin=True) as ax_2D:
        anim_24 = plot3d_red(ax_2D, HAND_POSE, HAND_SKELETON)
        anim_24.save('openpifpaf/plugins/apollocar3d/docs/CAR_24_Pose.gif', fps=30)


if __name__ == '__main__':
    main()
