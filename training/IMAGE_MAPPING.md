# Image Mapping — what each task feeds the model

Authoritative reference for the EOLLM vision SFT pipeline, generated from
`training/data.py` `convert_record()` + verified against the dataset records
(`splits_per_city/train.jsonl`). This is the contract the smoke test validates.

## Per-topic image mapping

| Topic | image_mode | #img | Images fed to model | SV labels |
|---|---|---|---|---|
| amenity_richness | satellite_marked | **5** | marked-sat + 4 SV | Fwd/Right/Bwd/Left |
| building_height | satellite_marked | **5** | marked-sat + 4 SV | Fwd/Right/Bwd/Left |
| green_space | satellite_marked | **5** | marked-sat + 4 SV | Fwd/Right/Bwd/Left |
| junction_type | satellite_marked | **5** | marked-sat + 4 SV | Fwd/Right/Bwd/Left |
| land_use | satellite_marked | **5** | marked-sat + 4 SV | Fwd/Right/Bwd/Left |
| road_surface | satellite_marked | **5** | marked-sat + 4 SV | Fwd/Right/Bwd/Left |
| road_type | satellite_marked | **5** | marked-sat + 4 SV | Fwd/Right/Bwd/Left |
| transit_density | satellite_marked | **5** | marked-sat + 4 SV | Fwd/Right/Bwd/Left |
| urban_density | satellite_marked | **5** | marked-sat + 4 SV | Fwd/Right/Bwd/Left |
| camera_direction | satellite_arrow | 5 | query-SV + 4 arrow-sats | A/B/C/D |
| mismatch_binary_easy | streetview_composite / streetview_binary | 5 | marked-sat + 4 SV | A/B/C/D |
| mismatch_binary_hard | streetview_composite / streetview_binary | 5 | marked-sat + 4 SV | A/B/C/D |
| mismatch_mcq_easy | streetview_mega | 2 | marked-sat + 1 mega 2×2 SV grid | (grid) |
| mismatch_mcq_hard | streetview_mega | 2 | marked-sat + 1 mega 2×2 SV grid | (grid) |

## The fix (2026-06-15)

The 9 `satellite_marked` topics previously fed **only the marked satellite
(1 image)** — the 4 street-view images present in every record were silently
dropped. They now feed **marked-sat + all 4 SV angles (5 images)**. The SV
images are labeled by viewing DIRECTION (Fwd/Right/Bwd/Left), NOT A/B/C/D,
because for these topics A/B/C/D are the ANSWER CHOICES (e.g. "A. Cobblestone")
— letter-labeling the images would make the model conflate image-A with
answer-A.

## Why labels differ by mode

- **satellite_marked** → direction labels (Fwd/Right/Bwd/Left): the SV images
  are 4 viewing angles of one place; A/B/C/D are reserved for answer options.
- **mismatch_* (binary)** → A/B/C/D labels: the task references specific images
  ("which image matches"), so the letter binds the choice to the image.
- **satellite_arrow (camera_direction)** → A/B/C/D on the 4 arrow-satellites:
  the question asks which arrow-image shows the correct direction.
- **streetview_mega (mismatch_mcq)** → one composite grid; the grid cells are
  self-labeling, no separate corner labels.

## SV angle order

`STV_ANGLES = ["along_fwd", "cross_right", "along_bwd", "cross_left"]`
→ rendered order Fwd, Right, Bwd, Left. The benchmark evaluator
(`EzelinyumEvaluator/src/dataio.py`) was aligned to this exact order + the
marked satellite + direction labels, so train and benchmark inputs match.

## Notes on coverage

- `building_height` / `green_space` have fewer than 2000 records (1431 / 1500)
  — `building:levels` / green-space tags are sparser in some cities.
- `mismatch_binary_easy/hard` SPAN two modes (composite for match=True, binary
  for match=False); both feed 5 images, content differs (own vs negative SVs).
