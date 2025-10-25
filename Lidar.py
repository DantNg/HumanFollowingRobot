import math
import time
import serial

HEADER_LO = 0xAA
HEADER_HI = 0x55

def _u16_le(lo, hi):
    return (hi << 8) | lo

class Lidar:
    """
    Dùng:
      with Lidar("COM15", 230400, has_intensity=False, model="triangle") as ld:
          scan = ld.get_full_rotation_dict(total_timeout=2.5, min_coverage_deg=340)
          if scan is None:
              print("Không gom đủ 1 vòng trong timeout")
          else:
              for ang, dist in sorted(scan.items()):
                  print(f"{ang:7.2f}° -> {dist:8.2f} mm")
    """
    def __init__(self, port, baud=230400, *, has_intensity=False, model="triangle", io_timeout=0.2):
        self.port = port
        self.baud = baud
        self.has_intensity = has_intensity
        self.model = model.lower()
        self.ser = serial.Serial(port=port, baudrate=baud, timeout=io_timeout)

    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): 
        try: self.close()
        except: pass

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

    # ---------- low-level I/O ----------
    def _read_exact(self, n, deadline):
        buf = bytearray()
        while len(buf) < n and time.time() <= deadline:
            chunk = self.ser.read(n - len(buf))
            if chunk:
                buf.extend(chunk)
        return bytes(buf) if len(buf) == n else None

    def read_packet(self, timeout=1.0):
        """Đọc đúng 1 packet (PH, CT, LSN, FSA, LSA, CS, samples) trong khoảng timeout tổng."""
        deadline = time.time() + timeout
        s = self.ser
        while time.time() <= deadline:
            b = s.read(1)
            if not b or b[0] != HEADER_LO:
                continue
            b2 = s.read(1)
            if not b2 or b2[0] != HEADER_HI:
                continue
            rest = self._read_exact(8, deadline)
            if not rest: return None
            lsn = rest[1]
            stride = 3 if self.has_intensity else 2
            need = lsn * stride
            samples = self._read_exact(need, deadline)
            if not samples: return None
            pkt = bytearray(10 + need)
            pkt[0:2] = (HEADER_LO, HEADER_HI)
            pkt[2:10] = rest
            pkt[10:] = samples
            return bytes(pkt)
        return None

    # ---------- packet parsing (giữ nguyên công thức của bạn) ----------
    @staticmethod
    def parse_packet(buf, has_intensity=False, model="triangle"):
        if len(buf) < 10:
            return {'ok': False, 'err': 'packet too short'}
        if buf[0] != HEADER_LO or buf[1] != HEADER_HI:
            return {'ok': False, 'err': 'bad header'}

        ct  = buf[2]
        lsn = buf[3]
        fsa = _u16_le(buf[4], buf[5])
        lsa = _u16_le(buf[6], buf[7])
        cs_pkt = _u16_le(buf[8], buf[9])

        stride = 3 if has_intensity else 2
        expected_len = 10 + lsn * stride
        if len(buf) < expected_len:
            return {'ok': False, 'err': f'need {expected_len}B, got {len(buf)}B'}

        samples = memoryview(buf)[10:10 + lsn * stride]

        # checksum
        checksum = _u16_le(HEADER_LO, HEADER_HI)
        checksum ^= fsa
        if has_intensity:
            for i in range(0, len(samples), 3):
                intensity_b = samples[i]
                d_lo = samples[i+1]; d_hi = samples[i+2]
                checksum ^= intensity_b
                checksum ^= _u16_le(d_lo, d_hi)
        else:
            for i in range(0, len(samples), 2):
                d_lo = samples[i]; d_hi = samples[i+1]
                checksum ^= _u16_le(d_lo, d_hi)

        checksum ^= ((lsn << 8) | (ct & 0xFF))
        checksum ^= lsa
        if checksum != cs_pkt:
            return {'ok': False, 'err': f'checksum mismatch ({checksum:04X} != {cs_pkt:04X})'}

        # distances & intensity
        dist_mm = [0.0] * lsn
        intensity = ([0] * lsn) if has_intensity else None
        j = 0
        if has_intensity:
            for i in range(lsn):
                s0 = samples[j]; s1 = samples[j+1]; s2 = samples[j+2]; j += 3
                inten = ((s1 & 0x03) << 8) | s0
                d_raw = (s2 << 8) | s1
                dist_mm[i] = float(d_raw >> 2)  # triangle/tof giống nhau ở nhánh này theo doc bạn
                intensity[i] = inten
        else:
            if model.lower() == "triangle":
                for i in range(lsn):
                    d_raw = (samples[j+1] << 8) | samples[j]; j += 2
                    dist_mm[i] = d_raw / 4.0
            else:
                for i in range(lsn):
                    d_raw = (samples[j+1] << 8) | samples[j]; j += 2
                    dist_mm[i] = float(d_raw)

        # angles cấp 1
        angle_fsa = ((fsa >> 1) / 64.0)
        angle_lsa = ((lsa >> 1) / 64.0)
        diff = angle_lsa - angle_fsa
        if diff < 0: diff += 360.0
        denom = max(1, lsn - 1)
        angles = [(i * diff) / denom + angle_fsa for i in range(lsn)]

        # sửa góc cho triangle
        if model.lower() == "triangle":
            for i, d in enumerate(dist_mm):
                if d > 0:
                    ang_corr = math.atan(21.8 * (155.3 - d) / (155.3 * d))
                    a = angles[i] + math.degrees(ang_corr)
                    angles[i] = a - 360.0 if a >= 360.0 else a

        zero_packet = (ct & 0x01) == 0x01 and lsn == 1
        sf_hz = ((ct >> 1) / 10.0) if (ct >> 1) != 0 else None

        return {
            'ok': True,
            'zero_packet': zero_packet,
            'sf_hz': sf_hz,
            'angles_deg': angles,
            'dist_mm': dist_mm,
            'intensity': intensity
        }

    # ---------- tiện ích ----------
    @staticmethod
    def _accumulate(points, angles_deg, dists_mm, angle_round=None, prefer="min"):
        """
        Gộp điểm vào dict. angle_round (deg): None hoặc ví dụ 0.5/1.0.
        prefer: 'min' (khoảng cách nhỏ nhất), 'last', 'mean'
        """
        if angle_round is None:
            # mỗi angle là 1 key float (chấp nhận overwrite sau)
            for a, d in zip(angles_deg, dists_mm):
                if d > 0:  # bỏ 0/invalid
                    if a in points:
                        if prefer == "min":
                            points[a] = min(points[a], d)
                        elif prefer == "last":
                            points[a] = d
                        elif prefer == "mean":
                            points[a] = (points[a][0] + d, points[a][1] + 1) if isinstance(points[a], tuple) else (points[a] + d, 2)
                    else:
                        points[a] = d if prefer != "mean" else (d, 1)
        else:
            for a, d in zip(angles_deg, dists_mm):
                if d <= 0: continue
                a_bin = round(a / angle_round) * angle_round
                if prefer == "min":
                    if a_bin in points: points[a_bin] = min(points[a_bin], d)
                    else: points[a_bin] = d
                elif prefer == "last":
                    points[a_bin] = d
                elif prefer == "mean":
                    if a_bin in points:
                        s, c = points[a_bin]
                        points[a_bin] = (s + d, c + 1)
                    else:
                        points[a_bin] = (d, 1)

    @staticmethod
    def _finalize_points(points, prefer):
        if prefer != "mean":
            return points
        # convert (sum,count) -> mean
        out = {}
        for k, v in points.items():
            if isinstance(v, tuple):
                s, c = v
                out[k] = s / max(1, c)
            else:
                out[k] = v
        return out

    # ---------- API: gom đủ 1 vòng ----------
    def get_full_rotation_dict(
        self,
        *,
        total_timeout=2.0,
        per_packet_timeout=0.5,
        min_coverage_deg=340.0,
        wrap_hysteresis_deg=20.0,
        angle_round=1.0,         # gộp theo 1° (đổi None nếu muốn giữ nguyên)
        prefer="min",            # 'min' | 'last' | 'mean'
        max_range_mm=6000.0
    ):
        """
        Gom các packet liên tiếp thành 1 vòng quét (~360°) rồi trả về dict{angle_deg: dist_mm}.
        - Không vòng lặp vô hạn: có tổng timeout.
        - wrap detection: khi góc bắt đầu của packet mới < góc bắt đầu trước đó - wrap_hysteresis_deg.
        - min_coverage_deg: yêu cầu độ phủ góc tối thiểu để coi là 1 vòng.
        """
        deadline = time.time() + total_timeout
        first_pkt_start = None
        prev_pkt_start = None
        unwrapped_min = None
        unwrapped_max = None
        seen_wrap = False

        # dùng dict tích luỹ (bin/keep-min)
        acc = {}

        while time.time() <= deadline:
            pkt = self.read_packet(timeout=min(per_packet_timeout, max(0.0, deadline - time.time())))
            if pkt is None:
                continue

            res = self.parse_packet(pkt, has_intensity=self.has_intensity, model=self.model)
            if not res.get('ok') or res.get('zero_packet'):
                continue

            angles = res["angles_deg"]
            dists  = res["dist_mm"]

            # lọc range
            for i in range(len(dists)):
                if dists[i] <= 0 or dists[i] > max_range_mm:
                    dists[i] = 0.0  # bỏ

            if not angles:
                continue

            pkt_start = angles[0]
            # Phát hiện wrap giữa các packet
            if prev_pkt_start is not None:
                if pkt_start < (prev_pkt_start - wrap_hysteresis_deg):
                    seen_wrap = True

            prev_pkt_start = pkt_start
            if first_pkt_start is None:
                first_pkt_start = pkt_start

            # Cập nhật độ phủ (sử dụng min/max theo 0..360, chấp nhận wrap)
            # Ta lấy min,max theo tập hiện tại (kể cả sau wrap sẽ trải 2 cụm); điều kiện chấm dứt dựa vào seen_wrap + min_coverage.
            cur_min = min(angles)
            cur_max = max(angles)
            unwrapped_min = cur_min if unwrapped_min is None else min(unwrapped_min, cur_min)
            unwrapped_max = cur_max if unwrapped_max is None else max(unwrapped_max, cur_max)
            coverage1 = unwrapped_max - unwrapped_min  # theo một nhánh
            # coverage alternative: nếu đã wrap, coi như gần 360
            coverage = coverage1 if not seen_wrap else 360.0 - (first_pkt_start - pkt_start) % 360.0
            # Tích luỹ điểm
            self._accumulate(acc, angles, dists, angle_round=angle_round, prefer=prefer)

            # Điều kiện đã đủ 1 vòng: đã thấy wrap & độ phủ đủ lớn
            if seen_wrap and coverage >= min_coverage_deg:
                # finalize nếu mean
                out = self._finalize_points(acc, prefer)
                # chuẩn hoá key angle vào [0,360)
                normalized = {}
                for a, d in out.items():
                    na = a % 360.0
                    # nếu binning 1°, có thể 0° và 360° trùng; unify về 0..359.9
                    if na in normalized:
                        if prefer == "min":
                            normalized[na] = min(normalized[na], d)
                        elif prefer == "last":
                            normalized[na] = d
                        elif prefer == "mean":
                            normalized[na] = (normalized[na] + d) / 2.0
                    else:
                        normalized[na] = d
                return normalized

        # Hết thời gian mà chưa đủ 1 vòng
        return None


if __name__ == "__main__":
    with Lidar("COM15", 115200, has_intensity=False, model="triangle", io_timeout=0.2) as ld:
        scan = ld.get_full_rotation_dict(
            total_timeout=2.5,
            min_coverage_deg=340.0,
            wrap_hysteresis_deg=20.0,
            angle_round=1.0,
            prefer="min",
            max_range_mm=6000.0
        )
        if scan is None:
            print("❌ Không gom đủ 1 vòng trong timeout")
        else:
            print(f"✅ Lấy được {len(scan)} bin góc")
            for ang, dist in sorted(scan.items()):
                print(f"{ang:7.2f}° -> {dist:8.2f} mm")
