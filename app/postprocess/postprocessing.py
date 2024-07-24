# _*_ coding:utf-8 _*_
"""
@date: 2024/6/20
@filename: postprocess_model_outputs
"""
# code from https://github.com/JigsawStack/insanely-fast-whisper-api.git
import sys
import numpy as np
from app.api.models import SegmentSpeakerResponse


def merge_speaker_segments(segments):
    merged_segments = []
    current_speaker = None
    current_start = None
    current_end = None

    for segment in segments:
        if current_speaker is None:
            # 初始化当前说话者和时间段
            current_speaker = segment['label']
            current_start = segment['segment']['start']
            current_end = segment['segment']['end']
        elif current_speaker == segment['label']:
            # 如果当前说话者与上一个说话者相同，则合并时间段
            current_end = segment['segment']['end']
        else:
            # 如果不同，则保存当前合并后的段，并重置当前说话者和时间段
            merged_segments.append({
                'label': current_speaker,
                'segment': {'start': current_start, 'end': current_end}
            })
            current_speaker = segment['label']
            current_start = segment['segment']['start']
            current_end = segment['segment']['end']

    # 确保最后一个段被添加
    if current_speaker is not None:
        merged_segments.append({
            'label': current_speaker,
            'segment': {'start': current_start, 'end': current_end}
        })

    return merged_segments


def diarize_audio(diarization):
    segments = []
    for segment, track, label in diarization.itertracks(yield_label=True):
        segments.append(
            {
                "segment": {"start": segment.start, "end": segment.end},
                "track": track,
                "label": label,
            }
        )
    new_segments = []
    prev_segment = cur_segment = segments[0]

    for i in range(1, len(segments)):
        cur_segment = segments[i]

        # check if we have changed speaker ("label")
        if cur_segment["label"] != prev_segment["label"] and i < len(segments):
            # add the start/end times for the super-segment to the new list
            new_segments.append(
                {
                    "segment": {
                        "start": prev_segment["segment"]["start"],
                        "end": cur_segment["segment"]["start"],
                    },
                    "speaker": prev_segment["label"],
                }
            )
            prev_segment = segments[i]

    # add the last segment(s) if there was no speaker change
    new_segments.append(
        {
            "segment": {
                "start": prev_segment["segment"]["start"],
                "end": cur_segment["segment"]["end"],
            },
            "speaker": prev_segment["label"],
        }
    )
    # diarizer output may contain consecutive segments from the same speaker (e.g. {(0 -> 1, speaker_1), (1 -> 1.5, speaker_1), ...})
    # we combine these segments to give overall timestamps for each speaker's turn (e.g. {(0 -> 1.5, speaker_1), ...})

    return new_segments


def post_process_segments_and_transcripts(new_segments, transcript, group_by_speaker) -> list:
    # get the end timestamps for each chunk from the ASR output
    end_timestamps = np.array(
        [chunk["timestamp"][-1] if chunk["timestamp"][-1] is not None else sys.float_info.max for chunk in transcript])
    segmented_preds = []
    # align the diarizer timestamps and the ASR timestamps
    for segment in new_segments:
        # get the diarizer end timestamp
        end_time = segment["segment"]["end"]
        # find the ASR end timestamp that is closest to the diarizer's end timestamp and cut the transcript to here
        upto_idx = np.argmin(np.abs(end_timestamps - end_time))

        if group_by_speaker:
            segmented_preds.append(
                SegmentSpeakerResponse(speaker=segment["speaker"], text="".join(
                    [chunk["text"] for chunk in transcript[: upto_idx + 1]]
                ), start=transcript[0]["timestamp"][0], end=transcript[upto_idx]["timestamp"][1])

            )
        else:
            for i in range(upto_idx + 1):
                segmented_preds.append({"speaker": segment["speaker"], **transcript[i]})

        # crop the transcripts and timestamp lists according to the latest timestamp (for faster argmin)
        transcript = transcript[upto_idx + 1:]
        end_timestamps = end_timestamps[upto_idx + 1:]

        if len(end_timestamps) == 0:
            break

    return segmented_preds
