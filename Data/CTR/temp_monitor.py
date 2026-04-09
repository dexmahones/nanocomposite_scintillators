#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import sys
import time
from pathlib import Path

import serial
import serial.tools.list_ports

# Matplotlib is optional (only imported if --plot is used)
def _lazy_import_matplotlib():
    global plt
    import matplotlib.pyplot as plt
    return plt

def autodetect_port(preferred_substrings=("Arduino", "wchusbserial", "usbserial", "usbmodem")):
    """
    Try to find an Arduino-like port. Returns port name or None.
    """
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None

    # Prefer ports that look like Arduino/USB serial
    for p in ports:
        desc = f"{p.description} {p.hwid}".lower()
        if any(s.lower() in desc for s in preferred_substrings):
            return p.device

    # Fallback: if only one port exists, return it
    if len(ports) == 1:
        return ports[0].device

    return None

def open_serial(port, baud, timeout=2.0, reset_wait=2.0):
    """
    Open serial and give Arduino time to reset after DTR is toggled.
    """
    ser = serial.Serial(port, baudrate=baud, timeout=timeout)
    # Give the board time to reset & start printing
    time.sleep(reset_wait)
    # Flush any partial lines
    ser.reset_input_buffer()
    return ser

def parse_line_to_floats(line):
    """
    Accepts lines such as:
      "23.45"
      "23.45,24.01"
      "23.45 C"
    Returns list[float] or None if parse fails.
    """
    line = line.strip()
    if not line:
        return None
    # Remove common units/symbols
    for token in ("C", "c", "°", "deg", "Deg", "DegC", "°C"):
        line = line.replace(token, "")
    # Split by comma/space
    # Try comma first; if no comma, split by whitespace
    parts = line.split(",") if "," in line else line.split()
    vals = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        try:
            vals.append(float(p))
        except ValueError:
            # Ignore non-numeric tokens; if none numeric, we'll return None
            pass
    return vals if vals else None

def main():
    ap = argparse.ArgumentParser(description="Read temperature(s) from Arduino over serial.")
    ap.add_argument("--port", help="Serial port (e.g., COM5, /dev/ttyACM0). If omitted, auto-detect.")
    ap.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200).")
    ap.add_argument("--csv", type=Path, default=Path(f"temps_{dt.datetime.now():%Y%m%d_%H%M%S}.csv"),
                    help="CSV filename to save (default: temps_YYYYmmdd_HHMMSS.csv).")
    ap.add_argument("--plot", action="store_true", help="Show live plot.")
    ap.add_argument("--interval", type=float, default=0.0,
                    help="Read/plot loop sleep (s). 0 = as fast as lines arrive.")
    ap.add_argument("--reps", type=int, help="Number of reads to average over", default = 2)
    args = ap.parse_args()

    port = args.port or autodetect_port()
    if not port:
        print("Could not auto-detect a serial port. Please pass --port COMx (Windows) or /dev/ttyXXX.", file=sys.stderr)
        sys.exit(2)

    print(f"Connecting to {port} @ {args.baud} ...")
    try:
        ser = open_serial(port, args.baud)
    except Exception as e:
        print(f"Failed to open serial port {port}: {e}", file=sys.stderr)
        sys.exit(3)

    # CSV setup
    csv_path = args.csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    writer = None
    header_written = False

    # Plot setup (optional)
    if args.plot:
        plt = _lazy_import_matplotlib()
        plt.ion()
        fig, ax = plt.subplots()
        lines = []   # one Line2D per channel
        xs = []
        ys = []      # list of lists; ys[i] is series for channel i
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Temperature (°C)")
        ax.set_title("Arduino Temperature")
        fig.tight_layout()

    t0 = time.time()
    print("Streaming... (Ctrl+C to stop)")
    try:
        while True:
            vals = [0]
            for i in range(args.reps):
                raw = ser.readline().decode("utf-8", errors="ignore")
                if not raw:
                    # No data within timeout; keep polling
                    if args.interval:
                        time.sleep(args.interval)
                    continue

                temp = parse_line_to_floats(raw)
                vals[0]+=temp[0]/args.reps
                time.sleep(args.interval)

            if vals is None:
                # Could print noisy/debug text occasionally
                continue

            now = dt.datetime.now()
            elapsed = time.time() - t0

            # Initialize CSV header on first valid line
            if not header_written:
                colnames = ["timestamp_iso", "elapsed_s"] + [f"t{i+1}_C" for i in range(len(vals))]
                writer = csv.writer(csv_file)
                writer.writerow(colnames)
                header_written = True
                print(f"Detected {len(vals)} channel(s): {', '.join(colnames[2:])}")
                print(f"Writing CSV → {csv_path.resolve()}")

                if args.plot:
                    xs = []
                    ys = [[] for _ in vals]
                    lines = [ax.plot([], [], label=f"T{i+1}")[0] for i in range(len(vals))]
                    ax.legend()

            # Write a row
            row = [now.isoformat(), f"{elapsed:.3f}"] + [f"{v:.3f}" for v in vals]
            writer.writerow(row)
            csv_file.flush()

            # Update plot
            if args.plot:
                xs.append(elapsed)
                for i, v in enumerate(vals):
                    ys[i].append(v)
                    lines[i].set_data(xs, ys[i])
                ax.relim()
                ax.autoscale_view()
                plt.pause(0.001)

            if args.interval:
                time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        try:
            ser.close()
        except Exception:
            pass
        try:
            csv_file.close()
        except Exception:
            pass
        if args.plot:
            try:
                plt.ioff()
                plt.show()
            except Exception:
                pass

if __name__ == "__main__":
    main()
