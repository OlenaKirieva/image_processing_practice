from __future__ import annotations

"""Lab 01 (skeleton): filtering/convolution + FFT tools (spatial & frequency domain)."""

import argparse
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import numpy.typing as npt

BorderType = Literal["reflect", "constant", "wrap", "replicate"]


def conv2d(
    image: npt.NDArray[np.generic],
    kernel: npt.NDArray[np.generic],
    border: BorderType = "reflect",
) -> np.ndarray:
    """
    2D convolution for grayscale/color images (spatial-domain linear filtering).

    Args:
        image: `(H,W)` or `(H,W,C)` array (any numeric dtype; computed in `float32`).
        kernel: `(kH,kW)` 2D kernel (any numeric dtype).
        border: `"reflect" | "constant" | "wrap" | "replicate"`.

    Returns:
        `float32` array with the same shape as `image`.
    """

    image_float = image.astype(np.float32)
    kernel_float = kernel.astype(np.float32)

    border_map = {
        "reflect": cv2.BORDER_REFLECT,
        "constant": cv2.BORDER_CONSTANT,
        "wrap": cv2.BORDER_WRAP,
        "replicate": cv2.BORDER_REPLICATE,
    }

    cv2_border = border_map.get(border, cv2.BORDER_REFLECT)

    result = cv2.filter2D(
        src=image_float, ddepth=-1, kernel=kernel_float, borderType=cv2_border
    )

    return result


def make_gaussian_kernel(ksize: int, sigma: float) -> npt.NDArray[np.float32]:
    """
    Create a normalized 2D Gaussian kernel (sum ~ 1).

    Args:
        ksize: Positive odd kernel size.
        sigma: Standard deviation in pixels (> 0).

    Returns:
        `(ksize, ksize)` `float32` kernel.
    """

    kernel_1d = cv2.getGaussianKernel(ksize, sigma, ktype=cv2.CV_32F)
    kernel_2d = np.outer(kernel_1d, kernel_1d).astype(np.float32)
    kernel_2d /= kernel_2d.sum()

    return kernel_2d


def _clip_to_dtype_range(x: np.ndarray, dtype: np.dtype) -> np.ndarray:

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
    elif np.issubdtype(dtype, np.floating):
        info = np.finfo(dtype)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    x_clipped = np.clip(x, info.min, info.max)

    return x_clipped.astype(dtype)


def apply_gaussian_blur(
    image: npt.NDArray[np.generic], ksize: int, sigma: float
) -> np.ndarray:
    """
    Gaussian smoothing in the spatial domain (via `conv2d`).

    Args:
        image: `(H,W)` or `(H,W,C)` image.
        ksize: Positive odd kernel size.
        sigma: Standard deviation in pixels.

    Returns:
        Same shape/dtype as input.
    """

    kernel = make_gaussian_kernel(ksize, sigma)
    blurred = conv2d(image, kernel, border="reflect")
    result = _clip_to_dtype_range(blurred, image.dtype)

    return result


def apply_box_blur(image: npt.NDArray[np.generic], ksize: int) -> np.ndarray:
    """
    Box/mean blur using a `(ksize x ksize)` uniform kernel (via `conv2d`).

    Args:
        image: `(H,W)` or `(H,W,C)` image.
        ksize: Positive odd window size.

    Returns:
        Same shape/dtype as input.
    """
    kernel = np.ones((ksize, ksize), dtype=np.float32) / (ksize**2)
    blurred = conv2d(image, kernel, border="reflect")

    return _clip_to_dtype_range(blurred, image.dtype)


def apply_median_blur(image: npt.NDArray[np.generic], ksize: int) -> np.ndarray:
    """
    Median filter (best for salt-and-pepper noise).

    Args:
        image: `uint8` image (grayscale or color).
        ksize: Positive odd neighborhood size.

    Returns:
        Same shape/dtype as input.
    """

    if image.dtype != np.uint8:
        image_uint8 = _clip_to_dtype_range(image, np.uint8)
    else:
        image_uint8 = image

    result = cv2.medianBlur(image_uint8, ksize)

    return result.astype(image.dtype)


def add_salt_pepper_noise(
    image: npt.NDArray[np.generic],
    amount: float,
    salt_vs_pepper: float = 0.5,
    *,
    seed: int = 0,
) -> np.ndarray:
    """
    Add salt-and-pepper (impulse) noise (deterministic by `seed`).

    Args:
        image: Input image (any numeric dtype).
        amount: Fraction of pixels to corrupt in `[0, 1]`.
        salt_vs_pepper: Probability of "salt" among corrupted pixels.
        seed: RNG seed.

    Returns:
        Noised image with the same shape/dtype.
    """
    out = image.copy()
    rng = np.random.default_rng(seed)

    prob = rng.random(out.shape)
    out[prob < (amount * salt_vs_pepper)] = (
        255 if np.issubdtype(image.dtype, np.integer) else 1.0
    )
    out[(prob >= (amount * salt_vs_pepper)) & (prob < amount)] = 0

    return out


def add_gaussian_noise(
    image: npt.NDArray[np.generic], sigma: float, *, seed: int = 0
) -> np.ndarray:
    """
    Add zero-mean Gaussian noise (deterministic by `seed`).

    Args:
        image: Input image (any numeric dtype).
        sigma: Standard deviation (>= 0), in image intensity units.
        seed: RNG seed.

    Returns:
        Noised image with the same shape/dtype.
    """

    rng = np.random.default_rng(seed)

    noise = rng.normal(loc=0.0, scale=sigma, size=image.shape)
    noised_image = image.astype(np.float32) + noise

    return _clip_to_dtype_range(noised_image, image.dtype)


def sobel_edges(
    image: npt.NDArray[np.generic], ksize: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sobel gradients and magnitude (edge strength).

    Returns:
        `(gx, gy, magnitude)` as `float32` arrays of shape `(H, W)`.

    Args:
        image: Input image (converted to grayscale internally).
        ksize: Positive odd Sobel size.
    """

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    gray = gray.astype(np.float32)
    gx = cv2.Sobel(src=gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize)
    gy = cv2.Sobel(src=gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize)

    magnitude = cv2.magnitude(gx, gy)

    return gx, gy, magnitude


def laplacian_edges(image: npt.NDArray[np.generic], ksize: int = 3) -> np.ndarray:
    """
    Laplacian edge response (absolute value).

    Args:
        image: Input image (converted to grayscale internally).
        ksize: Positive odd aperture size.

    Returns:
        `float32` array `(H, W)` (non-negative).
    """

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    gray = gray.astype(np.float32)
    laplacian = cv2.Laplacian(src=gray, ddepth=cv2.CV_32F, ksize=ksize)
    magnitude = np.abs(laplacian)

    return magnitude


def fft2_image(image: npt.NDArray[np.generic]) -> np.ndarray:
    """
    Compute the 2D DFT using OpenCV, returning a 2-channel float32 spectrum.

    Returns:
        spectrum: (H, W, 2) float32 array where spectrum[...,0] is Re and spectrum[...,1] is Im.

    Args:
        image: Input image (grayscale or color). Converted to grayscale internally.
    """

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    gray_f32 = gray.astype(np.float32)

    return cv2.dft(gray_f32, flags=cv2.DFT_COMPLEX_OUTPUT)


def fftshift2(spectrum: npt.NDArray[np.floating]) -> np.ndarray:
    """
    Shift the zero-frequency component to the center.

    Args:
        spectrum: A 2D array `(H,W)` or a 3D array `(H,W,2)` (OpenCV DFT format).

    Returns:
        Spectrum with quadrants swapped so that DC is at the center.
    """

    shifted_spectrum = np.fft.fftshift(spectrum, axes=(0, 1))

    return shifted_spectrum


def magnitude_spectrum(
    spectrum: npt.NDArray[np.floating], log_scale: bool = True
) -> np.ndarray:
    """
    Convert a 2-channel OpenCV DFT spectrum into a magnitude image.

    Args:
        spectrum: OpenCV DFT output in shape `(H, W, 2)` with Re/Im channels.
        log_scale: If True, returns `log(1 + magnitude)` which is the standard way to
            visualize FFT spectra with large dynamic range.

    Returns:
        `float32` array of shape `(H, W)` with non-negative values.
    """
    mag = cv2.magnitude(spectrum[..., 0], spectrum[..., 1])

    if log_scale:
        mag = np.log(1.0 + mag)

    return mag.astype(np.float32)


def ideal_low_pass_filter(
    shape: tuple[int, int] | tuple[int, int, int], cutoff_radius: float
) -> np.ndarray:
    """
    Create an ideal (hard) low-pass frequency-domain mask.

    Args:
        shape: Target shape, typically the DFT spectrum shape `(H,W,2)` or `(H,W)`.
        cutoff_radius: Cutoff radius in pixels (in the frequency plane).

    Returns:
        A `float32` mask of shape `(H, W, 2)` suitable for elementwise multiplication
        with an OpenCV DFT spectrum.
    """

    h, w = shape[:2]

    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]

    dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    mask_2d = (dist_from_center <= cutoff_radius).astype(np.float32)

    mask_3d = np.zeros((h, w, 2), dtype=np.float32)
    mask_3d[..., 0] = mask_2d
    mask_3d[..., 1] = mask_2d

    return mask_3d


def ideal_high_pass_filter(
    shape: tuple[int, int] | tuple[int, int, int], cutoff_radius: float
) -> np.ndarray:
    """
    Create an ideal (hard) high-pass frequency-domain mask.

    This is defined as `1 - ideal_low_pass_filter(...)`.
    """

    lpf_mask = ideal_low_pass_filter(shape, cutoff_radius)
    hpf_mask = 1.0 - lpf_mask

    return hpf_mask.astype(np.float32)


def apply_frequency_filter(
    image: npt.NDArray[np.generic], filter_mask: npt.NDArray[np.floating]
) -> np.ndarray:
    """
    Filter an image in the frequency domain using an (H,W) or (H,W,2) mask.

    Args:
        image: Input image (grayscale or color). Converted to grayscale internally.
        filter_mask:
            Either:
            - `(H, W)` single-channel mask (will be broadcast to 2 channels), or
            - `(H, W, 2)` OpenCV-compatible 2-channel mask.

    Returns:
        Filtered spatial-domain image as `float32` of shape `(H, W)`.
    """

    spectrum = fft2_image(image)
    spectrum_shifted = np.fft.fftshift(spectrum, axes=(0, 1))

    if filter_mask.ndim == 2:
        mask_2channel = np.stack([filter_mask, filter_mask], axis=-1)
    else:
        mask_2channel = filter_mask

    filtered_spectrum_shifted = spectrum_shifted * mask_2channel
    filtered_spectrum = np.fft.ifftshift(filtered_spectrum_shifted, axes=(0, 1))
    filtered_image = cv2.idft(filtered_spectrum)

    result = cv2.magnitude(filtered_image[..., 0], filtered_image[..., 1])

    return result.astype(np.float32)


def normalize_to_uint8(x: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    """
    Min-max normalize an array to `[0, 255]` (`uint8`) for visualization.

    Args:
        x: Any numeric array-like.

    Returns:
        2D/3D array (same shape as input) scaled to `uint8`.
    """

    x_arr = np.array(x, dtype=np.float32)
    x_min = np.min(x_arr)
    x_max = np.max(x_arr)

    if x_max == x_min:
        return np.zeros(x_arr.shape, dtype=np.uint8)

    x_norm = (x_arr - x_min) / (x_max - x_min) * 255.0

    return np.round(x_norm).astype(np.uint8)


def main() -> int:
    """
    Lab 01 demo (skeleton).

    Expected behavior after implementation:
    - load 1-2 images from `./imgs/`
    - synthesize salt&pepper + Gaussian noise
    - compare median vs Gaussian vs box blur
    - compute Sobel/Laplacian edges
    - visualize FFT magnitude spectrum and apply ideal LPF/HPF
    - save all outputs into `./out/lab01/` (no GUI windows)
    """
    parser = argparse.ArgumentParser(
        description="Lab 01 skeleton (implement functions first)."
    )
    parser.add_argument(
        "--img1", type=str, default="lenna.png", help="First image from ./imgs/"
    )
    parser.add_argument(
        "--img2", type=str, default="airplane.bmp", help="Second image from ./imgs/"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="out/lab01",
        help="Output directory (relative to repo root)",
    )
    args = parser.parse_args()

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    def normalize_to_uint8(x: npt.ArrayLike) -> npt.NDArray[np.uint8]:
        arr = np.asarray(x, dtype=np.float32)
        mn, mx = float(np.min(arr)), float(np.max(arr))
        if mx <= mn:
            return np.zeros_like(arr, dtype=np.uint8)
        y = (arr - mn) * (255.0 / (mx - mn))
        return np.clip(y, 0.0, 255.0).astype(np.uint8)

    def save_figure(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    repo_root = Path(__file__).resolve().parents[1]
    imgs_dir = repo_root / "imgs"
    out_dir = (repo_root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    img1 = cv2.imread(str(imgs_dir / args.img1), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(imgs_dir / args.img2), cv2.IMREAD_GRAYSCALE)
    if img1 is None:
        raise FileNotFoundError(str(imgs_dir / args.img1))
    if img2 is None:
        raise FileNotFoundError(str(imgs_dir / args.img2))

    missing: list[str] = []

    # --- Noise + denoise comparisons ---
    try:
        sp_noisy = add_salt_pepper_noise(img1, amount=0.08, salt_vs_pepper=0.55, seed=0)
        g_noisy = add_gaussian_noise(img1, sigma=15.0, seed=0)

        median_sp = apply_median_blur(sp_noisy, 5)
        gauss_sp = apply_gaussian_blur(sp_noisy, 5, 1.2)
        box_sp = apply_box_blur(sp_noisy, 5)

        gauss_g = apply_gaussian_blur(g_noisy, 5, 1.2)
        box_g = apply_box_blur(g_noisy, 5)

        plt.figure(figsize=(12, 6))
        for i, (title, im) in enumerate(
            [
                ("Original", img1),
                ("Salt & pepper", sp_noisy),
                ("Median (5x5)", median_sp),
                ("Gaussian (5,σ=1.2)", gauss_sp),
                ("Box (5x5)", box_sp),
                ("Gaussian noise", g_noisy),
            ],
            start=1,
        ):
            plt.subplot(2, 3, i)
            plt.title(title)
            plt.imshow(im, cmap="gray")
            plt.axis("off")
        save_figure(out_dir / "denoise_sp_and_examples.png")

        plt.figure(figsize=(12, 4))
        for i, (title, im) in enumerate(
            [
                ("Gaussian noise", g_noisy),
                ("Gaussian blur", gauss_g),
                ("Box blur", box_g),
            ],
            start=1,
        ):
            plt.subplot(1, 3, i)
            plt.title(title)
            plt.imshow(im, cmap="gray")
            plt.axis("off")
        save_figure(out_dir / "denoise_gaussian_noise.png")
    except NotImplementedError as exc:
        missing.append(str(exc))

    # --- Edge detection ---
    try:
        gx, gy, mag = sobel_edges(img2, ksize=3)
        _ = (gx, gy)
        lap = laplacian_edges(img2, ksize=3)

        plt.figure(figsize=(12, 4))
        for i, (title, im) in enumerate(
            [
                ("Input", img2),
                ("Sobel magnitude", normalize_to_uint8(mag)),
                ("Laplacian |·|", normalize_to_uint8(lap)),
            ],
            start=1,
        ):
            plt.subplot(1, 3, i)
            plt.title(title)
            plt.imshow(im, cmap="gray")
            plt.axis("off")
        save_figure(out_dir / "edges.png")

        cv2.imwrite(str(out_dir / "sobel_mag.png"), normalize_to_uint8(mag))
        cv2.imwrite(str(out_dir / "laplacian_abs.png"), normalize_to_uint8(lap))
    except NotImplementedError as exc:
        missing.append(str(exc))

    # --- FFT + frequency-domain filtering ---
    try:
        spec = fft2_image(img2)
        spec_shift = fftshift2(spec)
        mag = magnitude_spectrum(spec_shift, log_scale=True)

        lp = ideal_low_pass_filter(spec_shift.shape, cutoff_radius=30.0)
        hp = ideal_high_pass_filter(spec_shift.shape, cutoff_radius=30.0)
        lowpassed = apply_frequency_filter(img2, lp)
        highpassed = apply_frequency_filter(img2, hp)

        plt.figure(figsize=(12, 4))
        for i, (title, im) in enumerate(
            [
                ("Input", img2),
                ("Magnitude spectrum (log)", normalize_to_uint8(mag)),
                ("LPF result", normalize_to_uint8(lowpassed)),
                ("HPF result", normalize_to_uint8(highpassed)),
            ],
            start=1,
        ):
            plt.subplot(1, 4, i)
            plt.title(title)
            plt.imshow(im, cmap="gray")
            plt.axis("off")
        save_figure(out_dir / "fft_frequency_filters.png")
    except NotImplementedError as exc:
        missing.append(str(exc))

    if missing:
        (out_dir / "STATUS.txt").write_text(
            "Lab 01 demo is incomplete. Implement the TODO functions in labs/lab01_filtering_convolution_fft.py.\n\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {out_dir / 'STATUS.txt'}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
