"""EEG acquisition: BrainFlow boards and Lab Streaming Layer. Each source exposes fs, channels, read(), close()."""
COMMON_BOARDS = [
    ("SYNTHETIC_BOARD", "Synthetic (no hardware)"),
    ("MUSE_2_BOARD", "Muse 2 (Bluetooth)"),
    ("MUSE_S_BOARD", "Muse S (Bluetooth)"),
    ("MUSE_S_ATHENA_BOARD", "Muse S Athena (Bluetooth)"),
    ("MUSE_2016_BOARD", "Muse 2016 (Bluetooth)"),
    ("CYTON_BOARD", "OpenBCI Cyton 8-ch (USB dongle)"),
    ("CYTON_DAISY_BOARD", "OpenBCI Cyton + Daisy 16-ch"),
    ("GANGLION_BOARD", "OpenBCI Ganglion 4-ch (USB dongle)"),
    ("GANGLION_NATIVE_BOARD", "OpenBCI Ganglion (native Bluetooth)"),
    ("CROWN_BOARD", "Neurosity Crown"),
    ("UNICORN_BOARD", "g.tec Unicorn"),
    ("BRAINBIT_BOARD", "BrainBit"),
    ("ENOPHONE_BOARD", "Enophone"),
    ("NOTION_2_BOARD", "Neurosity Notion 2"),
    ("NOTION_1_BOARD", "Neurosity Notion 1"),
    ("CYTON_WIFI_BOARD", "OpenBCI Cyton + WiFi Shield"),
    ("CYTON_DAISY_WIFI_BOARD", "OpenBCI Cyton + Daisy + WiFi Shield"),
    ("GANGLION_WIFI_BOARD", "OpenBCI Ganglion + WiFi Shield"),
    ("GALEA_BOARD", "OpenBCI Galea"),
    ("MUSE_2_BLED_BOARD", "Muse 2 (BLED112 dongle)"),
    ("MUSE_S_BLED_BOARD", "Muse S (BLED112 dongle)"),
    ("MUSE_2016_BLED_BOARD", "Muse 2016 (BLED112 dongle)"),
    ("BRAINBIT_BLED_BOARD", "BrainBit (BLED112 dongle)"),
    ("CALLIBRI_EEG_BOARD", "Callibri EEG"),
    ("EXPLORE_4_CHAN_BOARD", "Mentalab Explore 4-ch"),
    ("EXPLORE_8_CHAN_BOARD", "Mentalab Explore 8-ch"),
    ("EXPLORE_PLUS_8_CHAN_BOARD", "Mentalab Explore Pro 8-ch"),
    ("EXPLORE_PLUS_32_CHAN_BOARD", "Mentalab Explore Pro 32-ch"),
    ("ANT_NEURO_EE_410_BOARD", "ANT Neuro eego EE-410"),
    ("ANT_NEURO_EE_411_BOARD", "ANT Neuro eego EE-411"),
    ("ANT_NEURO_EE_430_BOARD", "ANT Neuro eego EE-430"),
    ("ANT_NEURO_EE_225_BOARD", "ANT Neuro eego EE-225"),
    ("ANT_NEURO_EE_511_BOARD", "ANT Neuro eego EE-511"),
    ("FREEEEG32_BOARD", "FreeEEG32"),
    ("FREEEEG128_BOARD", "FreeEEG128"),
    ("PIEEG_BOARD", "PiEEG (Raspberry Pi)"),
    ("NEUROPAWN_KNIGHT_BOARD", "NeuroPawn Knight"),
    ("SYNCHRONI_OCTO_8_CHANNELS_BOARD", "Synchroni Octo 8-ch"),
    ("BIOLISTENER_BOARD", "BioListener"),
    ("AAVAA_V3_BOARD", "AAVAA v3"),
]

# Default electrode names for boards whose BrainFlow metadata is generic or user-configurable.
DEFAULT_MONTAGES = {
    "CYTON_BOARD": ["Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"],
    "CYTON_DAISY_BOARD": ["Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2",
                          "F7", "F8", "F3", "F4", "T7", "T8", "P3", "P4"],
    "GANGLION_BOARD": ["Fp1", "Fp2", "O1", "O2"],
    "GANGLION_NATIVE_BOARD": ["Fp1", "Fp2", "O1", "O2"],
}


def list_boards():
    try:
        from brainflow.board_shim import BoardIds
        names = {b.name for b in BoardIds}
    except ImportError:
        names = set()
    return [{"id": k, "label": v, "available": k in names} for k, v in COMMON_BOARDS]


class BrainFlowSource:
    kind = "brainflow"

    def __init__(self, board, serial_port="", mac_address="", ip_address="", ip_port=0, channels=None):
        from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
        BoardShim.disable_board_logger()
        try:
            self.board_id = BoardIds[board].value
        except KeyError:
            raise ValueError(f"unknown board {board}")
        p = BrainFlowInputParams()
        p.serial_port = serial_port or ""
        p.mac_address = mac_address or ""
        p.ip_address = ip_address or ""
        p.ip_port = int(ip_port or 0)
        self.label = board
        self.board = BoardShim(self.board_id, p)
        self.rows = BoardShim.get_eeg_channels(self.board_id)
        self.fs = float(BoardShim.get_sampling_rate(self.board_id))
        names = list(channels or [])
        if len(names) != len(self.rows):
            names = DEFAULT_MONTAGES.get(board, [])
        if len(names) != len(self.rows):
            try:
                names = list(BoardShim.get_eeg_names(self.board_id))
            except Exception:  # noqa: BLE001 - BrainFlow raises for boards without names
                names = []
        if len(names) != len(self.rows):
            names = [f"Ch{i + 1}" for i in range(len(self.rows))]
        self.channels = names
        self.board.prepare_session()
        try:
            self.board.start_stream()
        except Exception:
            self.board.release_session()
            raise

    def read(self):
        d = self.board.get_board_data()
        if d.shape[1] == 0:
            return []
        return d[self.rows].T.round(3).tolist()

    def close(self):
        try:
            self.board.stop_stream()
        finally:
            self.board.release_session()


class LSLSource:
    kind = "lsl"

    def __init__(self, stream_name="", timeout=8.0):
        from pylsl import StreamInlet, resolve_byprop
        streams = (resolve_byprop("name", stream_name, timeout=timeout) if stream_name
                   else resolve_byprop("type", "EEG", timeout=timeout))
        if not streams:
            raise ValueError("no LSL EEG stream found")
        self.inlet = StreamInlet(streams[0], max_chunklen=0)
        info = self.inlet.info()
        self.label = info.name()
        self.fs = float(info.nominal_srate())
        if self.fs <= 0:
            raise ValueError("LSL stream has an irregular sampling rate; a fixed rate is required")
        n = info.channel_count()
        names, ch = [], info.desc().child("channels").child("channel")
        for _ in range(n):
            label = ch.child_value("label")
            if not label:
                break
            names.append(label)
            ch = ch.next_sibling()
        self.channels = names if len(names) == n else [f"Ch{i + 1}" for i in range(n)]

    def read(self):
        samples, _ = self.inlet.pull_chunk(timeout=0.0)
        return [[round(float(v), 3) for v in s] for s in samples]

    def close(self):
        self.inlet.close_stream()


def open_source(cfg):
    kind = cfg.get("kind", "brainflow")
    if kind == "brainflow":
        return BrainFlowSource(cfg.get("board", "SYNTHETIC_BOARD"), cfg.get("serial_port", ""),
                               cfg.get("mac_address", ""), cfg.get("ip_address", ""), cfg.get("ip_port", 0),
                               cfg.get("channels"))
    if kind == "lsl":
        return LSLSource(cfg.get("lsl_name", ""), float(cfg.get("timeout", 8.0)))
    raise ValueError(f"unknown device kind {kind!r}")
