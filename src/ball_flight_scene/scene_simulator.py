from __future__ import annotations

import plotly.graph_objects as go


class SceneSimulator:
    def __init__(self):
        self.cameras = []
        self.objects = []
        self.paths = []
        self.fig3d = None
        self.size = 14
        self.linewidth = 3
        self.layout = self._init_figure_layout()
        self.colors = ["red", "blue", "green", "cyan", "magenta", "grey", "orange", "yellow"]

    def _init_figure_layout(self):
        size = self.size
        return dict(
            scene=dict(
                xaxis=dict(title="X (m)", range=[-size / 2, size / 2]),
                yaxis=dict(title="Y (m)", range=[0, size]),
                zaxis=dict(title="Z (m)", range=[-size / 3, size + size / 5]),
                aspectmode="cube",
            ),
            scene_camera=dict(
                eye=dict(x=2, y=0.2, z=0),
                up=dict(x=0, y=1, z=0),
            ),
        )

    def add_camera(self, camera):
        self.cameras.append(camera)

    def add_object(self, obj):
        self.objects.append(obj)

    def add_path(self, path):
        self.paths.append(path)

    def _get_next_color(self):
        next_color = self.colors.pop(0)
        self.colors.append(next_color)
        return next_color

    def _render_paths(self, fig):
        for path in self.paths:
            path.draw(fig, color=self._get_next_color(), linewidth=self.linewidth)

    def _render_objects(self, fig):
        for obj in self.objects:
            obj.draw(fig, color=self._get_next_color(), linewidth=self.linewidth)

    def _render_cameras(self, fig):
        for camera in self.cameras:
            camera.draw(fig, color=self._get_next_color(), linewidth=self.linewidth)

    def render_3d_scene(self, show_cameras=True, renderer="browser"):
        self.fig3d = go.Figure()
        self._render_paths(self.fig3d)
        self._render_objects(self.fig3d)
        if show_cameras:
            self._render_cameras(self.fig3d)
        self.fig3d.update_layout(self.layout)
        self.fig3d.show(renderer=renderer)

    def change_layout(self, **kwargs):
        self.layout = {**kwargs}

    def render_projected_scene(self, renderer="browser"):
        for camera_idx, camera in enumerate(self.cameras):
            fig2d = go.Figure()
            for path in self.paths:
                coords_3d = path.get_3d_coordinates()
                coords_2d = camera.project(coords_3d)
                fig2d.add_trace(
                    go.Scatter(x=coords_2d[:, 0], y=coords_2d[:, 1], mode="lines", name=path.name)
                )
            for obj in self.objects:
                coords_3d = obj.get_3d_coordinates()
                coords_2d = camera.project(coords_3d)
                fig2d.add_trace(
                    go.Scatter(x=coords_2d[:, 0], y=coords_2d[:, 1], mode="lines", name=obj.name)
                )
            fig2d.update_layout(
                xaxis=dict(title="Image X (pixels)", range=[0, camera.dimensions[0]]),
                yaxis=dict(
                    title="Image Y (pixels)",
                    range=[0, camera.dimensions[1]],
                    scaleanchor="x",
                    scaleratio=1,
                ),
                yaxis_autorange="reversed",
                title=f"Projected Scene from Camera {camera_idx}",
            )
            fig2d.show(renderer=renderer)
