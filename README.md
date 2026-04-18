# Project Montage Phase 2: The Studio Floor (Video and Audio Synthesis Layer)

This project implements a parallel multi-agent infrastructure using LangGraph to convert structured story manifests into temporal audiovisual scenes.

## Architecture
- **Scene Parser Agent**: Formats JSON manifests into parallel executable branches.
- **Voice Synthesis Agent**: Leverages Hugging Face API (TTS) to generate character voice segments.
- **Video Generation Agent**: Leverages stock/API generation to create video chunks.
- **Face Swap Agent**: Simulates maintaining character identity mapping mapping.
- **Lip Sync Agent**: Final combinator Agent that merges audio waveforms with facial geometry.

## Setup
`pip install -r requirements.txt`

Make sure `.env` contains `HF_API_TOKEN`.

## Execution
`python main.py`
