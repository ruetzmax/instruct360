import cv2
import argparse


def extract_first_n_frames(input_video_path, output_video_path, n_frames):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    count = 0
    while count < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        count += 1

    cap.release()
    out.release()
    print(f"Saved {count} frames to {output_video_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract the first N frames from a video and save as a new video.")
    parser.add_argument('--input', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, required=True, help='Path to output video file')
    parser.add_argument('--n_frames', type=int, required=True, help='Number of frames to extract')
    args = parser.parse_args()

    extract_first_n_frames(args.input, args.output, args.n_frames)


if __name__ == "__main__":
    main()
