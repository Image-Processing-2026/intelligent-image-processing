# Module 2 annotation policy (v1)

This policy defines the independent ground truth used to evaluate Module 2.
It is deliberately separate from model output: MediaPipe, GroundingDINO and
MobileSAM output may help an annotator draft a shape, but may never be accepted
as ground truth without manual correction and review.

## Canonical images and masks

- Annotate the original canonical RGB image dimensions. Any EXIF orientation,
  crop or resize must be recorded before annotation; label masks use nearest-
  neighbour transforms only.
- Binary target masks are lossless PNG or `.npy`: background is `0`, target is
  `255`/`True`. Instance masks use stable positive integer IDs; `0` is
  background. Ignore areas are stored separately and excluded from metrics.
- Each case records image and mask SHA-256, source/release, split, target,
  prompt, GT kind, annotation origin, policy version and review state.

## Face-oval ground truth

`face_oval` means the geometric region enclosed by the visible head/face oval,
including eyes, nose and mouth. It is not a skin, hair or person mask.

- Exclude neck, torso and background. Hair and ears are excluded unless the
  project explicitly changes this policy version.
- For eyeglasses, include the face region behind the lens; do not extend the
  oval to a temple/arm. For partial occlusion by a hand or object, draw the
  intended oval only when its path is reasonably inferable; otherwise mark the
  uncertain section as ignore.
- Profile, strongly cropped and heavily occluded faces must be tagged
  `difficult`. If a credible oval cannot be drawn, do not create a positive
  mask merely to satisfy a case count.
- Store one polygon and one hard mask per face. A second reviewer checks the
  overlay at original resolution before `review_status=reviewed`.

## Semantic and negative ground truth

- Semantic targets use the declared class mapping from their source release;
  do not use `label > 0` for a multi-class label map.
- Instance datasets preserve instance IDs. Union masks may be derived for the
  public union metric but cannot replace per-instance labels.
- A negative case is `empty` only after a human has checked the whole image for
  the requested target. Ambiguous images are `difficult` or use ignore masks.

## Review states

`draft` is excluded from acceptance. `single_review` means creator review only
and is excluded when `--require-reviewed` is supplied. `reviewed` requires a
second person (or an explicitly recorded independent review procedure).
