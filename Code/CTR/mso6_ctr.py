import numpy as np
import pyvisa

# ========================
# CONFIGURATION
# ========================

VISA_ADDRESS = 'USB0::0x0699::0x052B::C010101::INSTR'
CHANNELS = ['CH1', 'CH2', 'CH3', 'CH4']
NUM_FRAMES = 1000
ENCODING = 'RIBinary'
WIDTH = 2  # 2 bytes per sample

# ========================
# CONNECT TO SCOPE
# ========================

rm = pyvisa.ResourceManager()
scope = rm.open_resource(VISA_ADDRESS)
VISA_ADDRESS = rm.list_resources()[0]
scope.timeout = 30000  # 30 seconds to avoid timeouts for big transfers

print("Connected to:", scope.query('*IDN?').strip())

# Clear previous status
scope.write('*CLS')

# ========================
# FASTFRAME SETUP
# ========================
# Manual command: HORizontal:FASTframe:STATE ON
scope.write('HORizontal:FASTframe:STATE ON')

# Manual command: HORizontal:FASTframe:COUNt <NR1>
scope.write(f'HORizontal:FASTframe:COUNt {NUM_FRAMES}')

# Make sure it stops after all frames
scope.write('ACQuire:STOPAfter SEQUENCE')

print(f"FastFrame enabled with {NUM_FRAMES} frames.")

# ========================
# ARM ACQUISITION
# ========================
# Manual command: ACQuire:STATE ON
scope.write('ACQuire:STATE ON')

# Wait until complete
# Manual command: *OPC?
scope.query('*OPC?')
print("Acquisition complete.")

# ========================
# CHECK HOW MANY FRAMES WERE ACTUALLY ACQUIRED
# Manual command: ACQuire:NUMFRAMESACQuired?
frames_acquired = int(scope.query('ACQuire:NUMFRAMESACQuired?').strip())
print(f"Scope reports {frames_acquired} frames acquired.")

# ========================
# WAVEFORM TRANSFER SETUP
# ========================
# Encoding and width
scope.write(f'DATA:ENCdg {ENCODING}')
scope.write(f'DATA:WIDTH {WIDTH}')
scope.write('WAVeform:POINts:MODE RAW')

# We'll assume same record length for all channels
scope.write(f'DATA:SOURce {CHANNELS[0]}')
record_length = int(scope.query('WAVeform:POINts?').strip())
print(f"Record length per frame: {record_length} points")

# ========================
# FETCH PER-CHANNEL SCALING
# ========================
# Manual command section: WFMPre:YMULT? etc.
scaling_params = {}
for ch in CHANNELS:
    scope.write(f'DATA:SOURce {ch}')
    ymult = float(scope.query('WFMPRE:YMULT?'))
    yoff  = float(scope.query('WFMPRE:YOFF?'))
    yzero = float(scope.query('WFMPRE:YZERO?'))
    scaling_params[ch] = (ymult, yoff, yzero)
    print(f"{ch} scaling - YMULT: {ymult}, YOFF: {yoff}, YZERO: {yzero}")

# ========================
# ALLOCATE STORAGE
# ========================
num_channels = len(CHANNELS)
all_waveforms = np.zeros((frames_acquired, num_channels, record_length), dtype=float)

# ========================
# DOWNLOAD ALL FRAMES
# ========================
print("Starting waveform transfer...")

for frame in range(1, frames_acquired + 1):
    # Manual command: DATa:FRAMESTARt <NR1>
    scope.write(f'DATa:FRAMESTARt {frame}')
    scope.write(f'DATa:FRAMESTOP {frame}')

    for ch_idx, ch in enumerate(CHANNELS):
        # Manual command: DATa:SOURce <channel>
        scope.write(f'DATA:SOURce {ch}')

        # Manual command: CURVe?
        raw_data = scope.query_binary_values('CURVe?', datatype='h', container=np.array)

        # Apply channel-specific scaling
        ymult, yoff, yzero = scaling_params[ch]
        volts = (raw_data - yoff) * ymult + yzero

        all_waveforms[frame - 1, ch_idx, :] = volts

    if frame % 100 == 0 or frame == frames_acquired:
        print(f"Downloaded frame {frame}/{frames_acquired}")

print("Waveform transfer complete.")

# ========================
# SAVE DATA
# ========================
np.save('fastframe_4ch_waveforms.npy', all_waveforms)
print("Saved to fastframe_4ch_waveforms.npy")

# ========================
# CLEANUP
# ========================
scope.close()
rm.close()
print("Done.")
