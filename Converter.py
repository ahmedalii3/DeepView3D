from pathlib import Path

import numpy as np
from PIL import Image


class Converter:
    def __init__(
        self,
        file_path=None,
        focal_length=None,
        baseline=None,
        fx=None,
        fy=None,
        cx=None,
        cy=None,
    ):
        self.file_path = Path(file_path).expanduser() if file_path is not None else None
        self.focal_length = focal_length
        self.baseline = baseline
        self.fx = fx if fx is not None else focal_length
        self.fy = fy if fy is not None else focal_length
        self.cx = cx
        self.cy = cy

    def _resolve_path(self, path_like):
        return Path(path_like).expanduser()

    def _get_input_path(self, provided_path, argument_name):
        chosen_path = provided_path if provided_path is not None else self.file_path
        if chosen_path is None:
            raise ValueError(
                f"{argument_name} is required, or set file_path when creating Converter."
            )
        return self._resolve_path(chosen_path)

    def _default_output_path(self, input_path, suffix):
        input_path = Path(input_path)
        return input_path.with_name(f"{input_path.stem}{suffix}.npy")

    def _load_array_file(self, file_path):
        path = self._resolve_path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        ext = path.suffix.lower()
        if ext == ".npy":
            return np.load(path)
        if ext == ".npz":
            data = np.load(path)
            if len(data.files) == 0:
                raise ValueError(f"No arrays found in npz file: {path}")
            return data[data.files[0]]
        if ext == ".csv":
            return np.loadtxt(path, delimiter=",")
        if ext == ".txt":
            return np.loadtxt(path)
        if ext == ".png":
            return np.asarray(Image.open(path))
        raise ValueError(
            f"Unsupported input format: {path.suffix}. Supported: .npy, .npz, .csv, .txt, .png"
        )

    def _save_array_file(self, arr, output_path):
        path = self._resolve_path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ext = path.suffix.lower()

        if ext == ".npy":
            np.save(path, arr)
        elif ext == ".csv":
            np.savetxt(path, arr, delimiter=",")
        elif ext == ".txt":
            np.savetxt(path, arr)
        elif ext == ".png":
            image_arr = np.asarray(arr)
            if image_arr.ndim != 2:
                raise ValueError("PNG output requires a 2D array.")
            if np.issubdtype(image_arr.dtype, np.floating):
                if np.any(image_arr < 0):
                    raise ValueError("PNG output does not support negative depth values.")
                image_arr = np.rint(image_arr).astype(np.uint16)
            Image.fromarray(image_arr).save(path)
        else:
            raise ValueError(
                f"Unsupported output format: {path.suffix}. Supported: .npy, .csv, .txt, .png"
            )
        return path

    def _resolve_stereo_parameters(self, focal_length, baseline):
        f = self.focal_length if focal_length is None else focal_length
        b = self.baseline if baseline is None else baseline
        if f is None or b is None:
            raise ValueError(
                "focal_length and baseline are required for depth/disparity conversion."
            )
        if f <= 0 or b <= 0:
            raise ValueError("focal_length and baseline must be positive.")
        return float(f), float(b)

    def _resolve_intrinsics(self, image_shape, intrinsics):
        if intrinsics is None:
            fx = self.fx
            fy = self.fy if self.fy is not None else self.fx
            cx = self.cx
            cy = self.cy
        else:
            fx = intrinsics.get("fx")
            fy = intrinsics.get("fy", fx)
            cx = intrinsics.get("cx")
            cy = intrinsics.get("cy")

        if fx is None or fy is None:
            raise ValueError("Camera intrinsics must include fx and fy.")

        h, w = image_shape
        if cx is None:
            cx = (w - 1) / 2.0
        if cy is None:
            cy = (h - 1) / 2.0
        return float(fx), float(fy), float(cx), float(cy)

    def Convert_Disparity_to_Depth(
        self,
        disparity_file=None,
        output_file=None,
        focal_length=None,
        baseline=None,
        invalid_value=0.0,
    ):
        """
        Convert one disparity file to one depth file and return output path.
        """
        f, b = self._resolve_stereo_parameters(focal_length, baseline)
        disparity_path = self._get_input_path(disparity_file, "disparity_file")
        disparity = np.asarray(self._load_array_file(disparity_path), dtype=np.float32)
        if disparity.ndim != 2:
            raise ValueError("disparity file must contain a 2D array.")

        depth = np.full(disparity.shape, invalid_value, dtype=np.float32)
        valid = disparity > 0
        depth[valid] = (f * b) / disparity[valid]

        in_path = disparity_path
        out_path = (
            self._resolve_path(output_file)
            if output_file is not None
            else self._default_output_path(in_path, "_depth")
        )
        return self._save_array_file(depth, out_path)

    def Convert_Depth_to_Disparity(
        self,
        depth_file=None,
        output_file=None,
        focal_length=None,
        baseline=None,
        invalid_value=0.0,
    ):
        """
        Convert one depth file to one disparity file and return output path.
        """
        f, b = self._resolve_stereo_parameters(focal_length, baseline)
        depth_path = self._get_input_path(depth_file, "depth_file")
        depth = np.asarray(self._load_array_file(depth_path), dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError("depth file must contain a 2D array.")

        disparity = np.full(depth.shape, invalid_value, dtype=np.float32)
        valid = depth > 0
        disparity[valid] = (f * b) / depth[valid]

        in_path = depth_path
        out_path = (
            self._resolve_path(output_file)
            if output_file is not None
            else self._default_output_path(in_path, "_disparity")
        )
        # return self._save_array_file(disparity, out_path)
        return disparity

    def convert_Depth_to_PointCloud(
        self,
        depth_file=None,
        output_file=None,
        intrinsics=None,
        depth_scale=1.0,
    ):
        """
        Convert one depth file to one point-cloud file (Nx3 array) and return output path.
        """
        depth_path = self._get_input_path(depth_file, "depth_file")
        depth = np.asarray(self._load_array_file(depth_path), dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError("depth file must contain a 2D array.")
        if depth_scale <= 0:
            raise ValueError("depth_scale must be positive.")

        h, w = depth.shape
        fx, fy, cx, cy = self._resolve_intrinsics((h, w), intrinsics)
        z = depth / float(depth_scale)

        valid = z > 0
        ys, xs = np.where(valid)
        z_vals = z[ys, xs]
        x_vals = (xs - cx) * z_vals / fx
        y_vals = (ys - cy) * z_vals / fy
        points = np.stack([x_vals, y_vals, z_vals], axis=1).astype(np.float32)

        in_path = depth_path
        out_path = (
            self._resolve_path(output_file)
            if output_file is not None
            else self._default_output_path(in_path, "_pointcloud")
        )
        return self._save_array_file(points, out_path)

    def convert_PointCloud_to_Depth(
        self,
        image_shape,
        pointcloud_file=None,
        output_file=None,
        intrinsics=None,
        fill_value=0.0,
    ):
        """
        Convert one point-cloud file (Nx3) to one depth file and return output path.
        """
        pointcloud_path = self._get_input_path(pointcloud_file, "pointcloud_file")
        points = np.asarray(self._load_array_file(pointcloud_path), dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError("point cloud file must contain an Nx3 (or NxM) array.")
        if len(image_shape) != 2:
            raise ValueError("image_shape must be a tuple/list: (height, width).")

        h, w = int(image_shape[0]), int(image_shape[1])
        fx, fy, cx, cy = self._resolve_intrinsics((h, w), intrinsics)

        depth_map = np.full((h, w), fill_value, dtype=np.float32)
        z_buffer = np.full((h, w), np.inf, dtype=np.float32)

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        valid = z > 0
        x = x[valid]
        y = y[valid]
        z = z[valid]

        u = np.round((x * fx / z) + cx).astype(np.int32)
        v = np.round((y * fy / z) + cy).astype(np.int32)
        in_frame = (u >= 0) & (u < w) & (v >= 0) & (v < h)

        u = u[in_frame]
        v = v[in_frame]
        z = z[in_frame]

        for ui, vi, zi in zip(u, v, z):
            if zi < z_buffer[vi, ui]:
                z_buffer[vi, ui] = zi
                depth_map[vi, ui] = zi

        in_path = pointcloud_path
        out_path = (
            self._resolve_path(output_file)
            if output_file is not None
            else self._default_output_path(in_path, "_depth")
        )
        return self._save_array_file(depth_map, out_path)


if __name__ == "__main__":
    # Example usage:
    converter = Converter(file_path="/Users/ahmed_ali/Documents/GitHub/DeepView3D/Dataset/NVIDIA/fat/mixed/kitchen_0/000000.left.depth.png", focal_length=768.16058349609375, baseline=0.1)
    intrinsics = 	{
				"fx": 768.16058349609375,
				"fy": 768.16058349609375,
				"cx": 480,
				"cy": 270,
				"s": 0
			}
    # converter.Convert_Disparity_to_Depth(disparity_file="disp.npy")
    # converter.Convert_Depth_to_Disparity(depth_file="/Users/ahmed_ali/Documents/GitHub/DeepView3D/Dataset/NVIDIA/fat/mixed/kitchen_0/000000.left.depth.png",baseline=6.0)
    # converter.convert_Depth_to_PointCloud(depth_file="/Users/ahmed_ali/Documents/GitHub/DeepView3D/Dataset/NVIDIA/fat/mixed/kitchen_0/000000.left.depth.png", intrinsics=intrinsics)
    # converter.convert_PointCloud_to_Depth(
    #     image_shape=(480, 640), pointcloud_file="pointcloud.npy"
    # )
