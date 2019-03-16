#! /usr/bin/env python3

__all__ = [
    'CameraTrackRenderer'
]

from typing import List, Tuple

import numpy as np
from OpenGL import GL
from OpenGL.GL import shaders
from OpenGL import GLUT
from OpenGL.arrays import vbo

import data3d


def _build_uncolored_elements_program():
    uncolored_elements_vertex_shader = shaders.compileShader(
        """
        #version 130
        uniform mat4 mvp;
        in vec3 position;

        void main() {
            vec4 camera_space_position = mvp * vec4(position, 1.0);
            gl_Position = camera_space_position;
        }""",
        GL.GL_VERTEX_SHADER
    )
    uncolored_elements_fragment_shader = shaders.compileShader(
        """
        #version 130
        uniform vec3 color;
        out vec3 out_color;

        void main() {
            out_color = color;
        }""",
        GL.GL_FRAGMENT_SHADER
    )

    return shaders.compileProgram(
        uncolored_elements_vertex_shader, uncolored_elements_fragment_shader
    )


def _build_point_cloud_program():
    point_cloud_vertex_shader = shaders.compileShader(
        """
        #version 130
        
        uniform mat4 mvp;
        in vec3 position;
        in vec3 in_color;
        out vec3 color;
        
        void main() {
            vec4 camera_space_position = mvp * vec4(position, 1.0);
            gl_Position = camera_space_position;
            color = in_color;
        }""",
        GL.GL_VERTEX_SHADER
    )
    point_cloud_fragment_shader = shaders.compileShader(
        """
        #version 130
        
        in vec3 color;
        out vec3 out_color;

        void main() {
            out_color = color;
        }""",
        GL.GL_FRAGMENT_SHADER
    )

    return shaders.compileProgram(
        point_cloud_vertex_shader, point_cloud_fragment_shader
    )


class CameraTrackRenderer:

    @staticmethod
    def _get_frustum(z_far, tracked_cam_parameters: data3d.CameraParameters):
        y_far = z_far * np.tan(tracked_cam_parameters.fov_y / 2)
        x_far = y_far * tracked_cam_parameters.aspect_ratio
        return np.array([[x_far * dx, y_far * dy, -z_far]
                         for (dx, dy) in [(-1, 1), (1, 1), (1, -1), (-1, -1)]], dtype=np.float32)

    def __init__(self,
                 cam_model_files: Tuple[str, str],
                 tracked_cam_parameters: data3d.CameraParameters,
                 tracked_cam_track: List[data3d.Pose],
                 point_cloud: data3d.PointCloud):
        """
        Initialize CameraTrackRenderer. Load camera model, create buffer objects, load textures,
        compile shaders, e.t.c.

        :param cam_model_files: path to camera model obj file and texture. The model consists of
        triangles with per-point uv and normal attributes
        :param tracked_cam_parameters: tracked camera field of view and aspect ratio. To be used
        for building tracked camera frustum
        :param point_cloud: colored point cloud
        """

        _FRUSTUM_Z_FAR = 20
        self._FRUSTUM_COLOR = np.array([1, 0, 0])
        self._PATH_COLOR = np.array([1, 1, 1])
        self._n_points = point_cloud.ids.shape[0]
        self._opencv2opengl = np.array(np.diag([1, -1, -1, 1]), dtype=np.float32)

        coords = np.array(point_cloud.points.reshape(-1), dtype=np.float32)
        colors = np.array(point_cloud.colors.reshape(-1), dtype=np.float32)
        self._point_cloud_coords_buffer = vbo.VBO(coords)
        self._point_cloud_colors_buffer = vbo.VBO(colors)

        self._camera_path_position = np.array([x.t_vec for x in tracked_cam_track], dtype=np.float32)
        self._camera_path_rotation = np.array([x.r_mat for x in tracked_cam_track], dtype=np.float32)
        self._camera_path_position_buffer = vbo.VBO(self._camera_path_position)

        frustum = self._get_frustum(_FRUSTUM_Z_FAR, tracked_cam_parameters)
        self._frustum_buffer = vbo.VBO(frustum)
        position = np.zeros(3, dtype=np.float32)
        self._frustum_segments_buffer = vbo.VBO(np.array([position, frustum[0],
                                                          position, frustum[1],
                                                          position, frustum[2],
                                                          position, frustum[3]], dtype=np.float32))

        self._uncolored_elements_program = _build_uncolored_elements_program()
        self._point_cloud_program = _build_point_cloud_program()

        GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA | GLUT.GLUT_DOUBLE | GLUT.GLUT_DEPTH)
        GL.glEnable(GL.GL_DEPTH_TEST)

    def display(self, camera_tr_vec, camera_rot_mat, camera_fov_y, tracked_cam_track_pos_float):
        """
        Draw everything with specified render camera position, projection parameters and 
        tracked camera position

        :param camera_tr_vec: vec3 position of render camera in global space
        :param camera_rot_mat: mat3 rotation matrix of render camera in global space
        :param camera_fov_y: render camera field of view. To be used for building a projection
        matrix. Use glutGet to calculate current aspect ratio
        :param tracked_cam_track_pos_float: a frame in which tracked camera
        model and frustum should be drawn (see tracked_cam_track_pos for basic task)
        :return: returns nothing
        """

        # a frame in which a tracked camera model and frustum should be drawn
        # without interpolation
        tracked_cam_track_pos = int(tracked_cam_track_pos_float)

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        aspect_ratio = GLUT.glutGet(GLUT.GLUT_WINDOW_WIDTH) / GLUT.glutGet(GLUT.GLUT_WINDOW_HEIGHT)

        model = np.eye(4, dtype=np.float32)
        view = self._get_view(camera_tr_vec, camera_rot_mat)
        projection = self._get_projection(camera_fov_y, aspect_ratio)

        mvp = projection.dot(view.dot(model.dot(self._opencv2opengl)))

        self._render_point_cloud(mvp)
        self._render_camera_path(mvp)
        self._render_frustum(mvp, self._camera_path_position[tracked_cam_track_pos],
                             self._camera_path_rotation[tracked_cam_track_pos])

        GLUT.glutSwapBuffers()

    @staticmethod
    def _get_view(camera_tr_vec, camera_rot_mat):
        translation = np.eye(4, dtype=np.float32)
        translation[:3, 3] = -camera_tr_vec

        rotation = np.eye(4, dtype=np.float32)
        rotation[:3, :3] = np.linalg.inv(camera_rot_mat)

        return rotation.dot(translation)

    @staticmethod
    def _get_projection(fovy, aspect_ratio, z_near=0.1, z_far=50):
        y_near = np.tan(fovy / 2) * z_near
        x_near = y_near * aspect_ratio
        fx = z_near / x_near
        fy = z_near / y_near
        a = -(z_far + z_near) / (z_far - z_near)
        b = -2 * z_far * z_near / (z_far - z_near)
        return np.array([[fx, 0, 0, 0],
                         [0, fy, 0, 0],
                         [0, 0, a, b],
                         [0, 0, -1, 0]],
                        dtype=np.float32)

    def _render_one_color_element(self, mvp, points_buffer, color, element_type):
        shaders.glUseProgram(self._uncolored_elements_program)

        GL.glUniformMatrix4fv(
            GL.glGetUniformLocation(self._uncolored_elements_program, 'mvp'),
            1, True, mvp)

        GL.glUniform3fv(
            GL.glGetUniformLocation(self._uncolored_elements_program, 'color'),
            1, color)

        points_buffer.bind()
        position_loc = GL.glGetAttribLocation(self._uncolored_elements_program, 'position')
        GL.glEnableVertexAttribArray(position_loc)
        GL.glVertexAttribPointer(position_loc, 3, GL.GL_FLOAT,
                                 False, 0,
                                 points_buffer)
        GL.glDrawArrays(element_type, 0, points_buffer.shape[0])

        GL.glDisableVertexAttribArray(position_loc)
        points_buffer.unbind()
        shaders.glUseProgram(0)

    def _render_camera_path(self, mvp):
        self._render_one_color_element(mvp, self._camera_path_position_buffer, self._PATH_COLOR, GL.GL_LINE_STRIP)

    def _render_frustum(self, mvp, position, rotation):
        translation_mat = np.eye(4, dtype=np.float32)
        translation_mat[:3, 3] = position

        rotation_mat = np.eye(4, dtype=np.float32)
        rotation_mat[:3, :3] = rotation
        rotation_mat = rotation_mat.dot(self._opencv2opengl)

        mvp = mvp.dot(translation_mat.dot(rotation_mat))

        self._render_one_color_element(mvp, self._frustum_buffer, self._FRUSTUM_COLOR, GL.GL_LINE_LOOP)
        self._render_one_color_element(mvp, self._frustum_segments_buffer, self._FRUSTUM_COLOR, GL.GL_LINES)

    def _render_point_cloud(self, mvp):
        shaders.glUseProgram(self._point_cloud_program)

        GL.glUniformMatrix4fv(
            GL.glGetUniformLocation(self._point_cloud_program, 'mvp'),
            1, True, mvp)

        self._point_cloud_coords_buffer.bind()
        position_loc = GL.glGetAttribLocation(self._point_cloud_program, 'position')
        GL.glEnableVertexAttribArray(position_loc)
        GL.glVertexAttribPointer(position_loc, 3, GL.GL_FLOAT,
                                 False, 0,
                                 self._point_cloud_coords_buffer)

        self._point_cloud_colors_buffer.bind()
        color_loc = GL.glGetAttribLocation(self._point_cloud_program, 'in_color')
        GL.glEnableVertexAttribArray(color_loc)
        GL.glVertexAttribPointer(color_loc, 3, GL.GL_FLOAT,
                                 False, 0,
                                 self._point_cloud_colors_buffer)

        GL.glDrawArrays(GL.GL_POINTS, 0, self._n_points)

        GL.glDisableVertexAttribArray(position_loc)
        GL.glDisableVertexAttribArray(color_loc)

        self._point_cloud_coords_buffer.unbind()
        self._point_cloud_colors_buffer.unbind()

        shaders.glUseProgram(0)
