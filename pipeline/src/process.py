import os
import math
import gzip
import orjson
import pickle
import xarray as xr
import numpy as np
from PIL import Image
import eccodes as ecc
from scipy.interpolate import LinearNDInterpolator
from pmtiles.writer import Writer
from pmtiles.tile import TileType, Compression, zxy_to_tileid
import time


def tile_bounds_wgs84(z, x, y):
    """Berechnet die exakte Bounding-Box (lon_min, lat_min, lon_max, lat_max)
    einer Web-Mercator Kachel in WGS84 Grad.
    """
    n = 2.0 ** z
    lon_min = x / n * 360.0 - 180.0
    lon_max = (x + 1) / n * 360.0 - 180.0

    lat_rad_max = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))
    lat_rad_min = math.atan(math.sinh(math.pi * (1.0 - 2.0 * (y + 1) / n)))

    return lon_min, math.degrees(lat_rad_min), lon_max, math.degrees(lat_rad_max)


class WindProcessor:
    def __init__(self, root_folder, timeLineLength, output_folder="./wind_tiles_simulation"):
        init_start_time = time.perf_counter()
        self.output_folder = output_folder
        self.cluster_output_folder = os.path.join(output_folder, "grid_cluster")
        os.makedirs(self.cluster_output_folder, exist_ok=True)
        self.root_folder = root_folder
        self.timeLineLength = timeLineLength
        
        self.cluster_memory = {}

        clat_path = os.path.join(self.root_folder, "clat.grib2")
        clon_path = os.path.join(self.root_folder, "clon.grib2")

        if not os.path.exists(clat_path) or not os.path.exists(clon_path):
            raise FileNotFoundError(f"Geometry files missing in folder: {self.root_folder}")

        with open(clat_path, "rb") as f:
            gid = ecc.codes_grib_new_from_file(f)
            raw_lat = ecc.codes_get_array(gid, "values")
            ecc.codes_release(gid)

        with open(clon_path, "rb") as f:
            gid = ecc.codes_grib_new_from_file(f)
            raw_lon = ecc.codes_get_array(gid, "values")
            ecc.codes_release(gid)

        if max(abs(raw_lat)) < 7:
            lat_deg = raw_lat * (180.0 / np.pi)
            lon_deg = raw_lon * (180.0 / np.pi)
        else:
            lat_deg = raw_lat
            lon_deg = raw_lon
        lon_deg = np.where(lon_deg > 180, lon_deg - 360, lon_deg)

        self.lon_min, self.lon_max = -4.1616, 20.5444
        self.lat_min, self.lat_max = 43.0440, 58.1647

        lat_pts_rad = np.radians(lat_deg)
        y_pts_merc = np.degrees(np.log(np.tan(np.pi/4.0 + lat_pts_rad/2.0)))
        x_pts_merc = lon_deg

        y_min_merc = np.degrees(np.log(np.tan(np.pi/4.0 + np.radians(self.lat_min)/2.0)))
        y_max_merc = np.degrees(np.log(np.tan(np.pi/4.0 + np.radians(self.lat_max)/2.0)))

        self.width = 2000
        self.height = int(self.width * (y_max_merc - y_min_merc) / (self.lon_max - self.lon_min))

        dx_pixel = (self.lon_max - self.lon_min) / self.width
        dy_pixel = (y_max_merc - y_min_merc) / self.height

        grid_x_linear = np.linspace(self.lon_min + 0.5 * dx_pixel, self.lon_max - 0.5 * dx_pixel, self.width)
        grid_y_merc = np.linspace(y_max_merc - 0.5 * dy_pixel, y_min_merc + 0.5 * dy_pixel, self.height)
        
        self.grid_x, self.grid_y = np.meshgrid(grid_x_linear, grid_y_merc)

        points_merc = np.vstack((x_pts_merc, y_pts_merc)).T.astype(np.float64)
        self.total_dwd_points = len(points_merc)

        self.interpolator = LinearNDInterpolator(points_merc, np.zeros(self.total_dwd_points, dtype=np.float64))

        # ---------------------------------------------------------------------
        # PMTILES GRID (Reguläres WGS84 0.02° Lat/Lon Raster)
        # ---------------------------------------------------------------------
        self.step_size = 0.02
        self.pm_lats = np.arange(self.lat_min, self.lat_max + self.step_size, self.step_size)
        self.pm_lons = np.arange(self.lon_min, self.lon_max + self.step_size, self.step_size)
        self.src_lat_shape = len(self.pm_lats)
        self.src_lon_shape = len(self.pm_lons)

        pm_grid_lon, pm_grid_lat = np.meshgrid(self.pm_lons, self.pm_lats)
        pm_points_wgs84 = np.vstack((lon_deg, lat_deg)).T.astype(np.float64)

        self.pm_interpolator_u = LinearNDInterpolator(pm_points_wgs84, np.zeros(self.total_dwd_points, dtype=np.float64))
        self.pm_interpolator_v = LinearNDInterpolator(pm_points_wgs84, np.zeros(self.total_dwd_points, dtype=np.float64))
        self.pm_grid_lon = pm_grid_lon
        self.pm_grid_lat = pm_grid_lat

        cluster_cols = np.floor((lon_deg - self.lon_min) / 1.0).astype(np.int32)
        cluster_rows = np.floor((lat_deg - self.lat_min) / 1.0).astype(np.int32)

        unique_clusters = np.unique(np.column_stack((cluster_cols, cluster_rows)), axis=0)

        self.cluster_mapping = {}
        for col, row in unique_clusters:
            point_indices = np.where((cluster_cols == col) & (cluster_rows == row))[0]
            self.cluster_mapping[(int(col), int(row))] = {
                "indices": point_indices,
                "lats": np.round(lat_deg[point_indices], 4).tolist(),
                "lons": np.round(lon_deg[point_indices], 4).tolist()
            }

        scipy_warmup_start_init = time.perf_counter()
        dummy_input_warmup = np.zeros((self.total_dwd_points, 1), dtype=np.float64)
        self.interpolator.values[:, 0] = dummy_input_warmup[:, 0]
        _ = self.interpolator(self.grid_x, self.grid_y)
        scipy_warmup_duration_init = time.perf_counter() - scipy_warmup_start_init
        print(f"    ⏱️ SciPy Interpolator Warmup (in init): {scipy_warmup_duration_init:.4f}s")

        init_duration = time.perf_counter() - init_start_time
        print(f"✅ [Processor] Gitter reaktiviert: {self.width}x{self.height} Pixel | {len(self.cluster_mapping)} Cluster bereit. (Init time: {init_duration:.4f}s)")

    def _create_wind_direction_pmtiles(self, u_grid_002, v_grid_002, output_pmtiles_path, min_zoom=0, max_zoom=8):
        """Erzeugt binäre Float32 PMTiles mit Gzip-Komprimierung und 24-Byte Header aus dem 0.02° Raster."""
        lats = self.pm_lats
        lons = self.pm_lons

        tiles_dict = {}

        for z in range(min_zoom, max_zoom + 1):
            stride = 2 ** (max_zoom - z)

            u_lod = u_grid_002[::stride, ::stride]
            v_lod = v_grid_002[::stride, ::stride]
            lats_lod = lats[::stride]
            lons_lod = lons[::stride]

            n = 2 ** z
            for x in range(n):
                for y in range(n):
                    t_lon_min, t_lat_min, t_lon_max, t_lat_max = tile_bounds_wgs84(z, x, y)

                    if (t_lon_max < self.lon_min or t_lon_min > self.lon_max or
                        t_lat_max < self.lat_min or t_lat_min > self.lat_max):
                        continue

                    col_indices = np.where((lons_lod >= t_lon_min) & (lons_lod <= t_lon_max))[0]
                    row_indices = np.where((lats_lod >= t_lat_min) & (lats_lod <= t_lat_max))[0]

                    if len(col_indices) == 0 or len(row_indices) == 0:
                        continue

                    c_start = col_indices[0]
                    c_end = col_indices[-1] + 1

                    r_start_idx = row_indices[-1]
                    r_end_idx = row_indices[0]

                    sub_u = u_lod[r_end_idx:r_start_idx + 1, c_start:c_end]
                    sub_v = v_lod[r_end_idx:r_start_idx + 1, c_start:c_end]

                    # Zeilen umkehren für North-Up
                    sub_u = np.flipud(sub_u)
                    sub_v = np.flipud(sub_v)

                    rows, cols = sub_u.shape
                    if rows == 0 or cols == 0:
                        continue

                    valid_mask = ~np.isnan(sub_u) & ~np.isnan(sub_v)
                    if not np.any(valid_mask):
                        continue

                    # 1. HEADER (6x Float32 = 24 Bytes)
                    origin_lng = float(lons_lod[c_start])
                    origin_lat = float(lats_lod[r_start_idx])

                    delta_lng = float(lons_lod[1] - lons_lod[0]) if len(lons_lod) > 1 else self.step_size * stride
                    delta_lat = float(lats_lod[1] - lats_lod[0]) if len(lats_lod) > 1 else self.step_size * stride

                    header_meta = np.array([
                        origin_lng,
                        origin_lat,
                        delta_lng,
                        delta_lat,
                        float(rows),
                        float(cols)
                    ], dtype=np.float32)

                    # 2. PAYLOAD (u, v verschachtelt als Float32)
                    uv_interleaved = np.empty((rows, cols, 2), dtype=np.float32)
                    uv_interleaved[:, :, 0] = sub_u
                    uv_interleaved[:, :, 1] = sub_v

                    raw_tile_bytes = header_meta.tobytes() + uv_interleaved.tobytes()
                    compressed_tile_bytes = gzip.compress(raw_tile_bytes)

                    tiles_dict[zxy_to_tileid(z, x, y)] = compressed_tile_bytes

        with open(output_pmtiles_path, "wb") as f:
            writer = Writer(f)

            for tile_id in sorted(tiles_dict.keys()):
                writer.write_tile(tile_id, tiles_dict[tile_id])

            header = {
                "tile_type": TileType.UNKNOWN,
                "tile_compression": Compression.GZIP,
                "min_zoom": min_zoom,
                "max_zoom": max_zoom,
                "min_lon": self.lon_min,
                "min_lat": self.lat_min,
                "max_lon": self.lon_max,
                "max_lat": self.lat_max,
                "center_zoom": 5,
                "center_lon": (self.lon_min + self.lon_max) / 2.0,
                "center_lat": (self.lat_min + self.lat_max) / 2.0
            }

            metadata = {
                "name": "DWD Wind Vector Binary PMTiles (0.02°)",
                "format": "binary",
                "description": "24 Byte Float32 Header + Interleaved Float32 (U, V) Payload mit Gzip."
            }

            writer.finalize(header, metadata)

        return True

    def process_step(self, u_path, v_path, gust_path, time_key, filename):
        step_start_time = time.perf_counter()

        if not os.path.exists(u_path) or not os.path.exists(v_path) or not os.path.exists(gust_path):
            print(f"⚠️ GRIB2-Dateien für {filename} unvollständig. Überspringe Verarbeitung.")
            return False

        print(f"-> Berechne Wind und interpoliere für {filename}...")

        grib_load_start = time.perf_counter()
        with open(u_path, 'rb') as f:
            gid_u = ecc.codes_grib_new_from_file(f)
            u_values = ecc.codes_get_array(gid_u, 'values')
            try:
                u_missing_value = ecc.codes_get(gid_u, 'missingValue')
                u_values[u_values == u_missing_value] = np.nan
            except ecc.CodesInternalError:
                pass
            ecc.codes_release(gid_u)

        with open(v_path, 'rb') as f:
            gid_v = ecc.codes_grib_new_from_file(f)
            v_values = ecc.codes_get_array(gid_v, 'values')
            try:
                v_missing_value = ecc.codes_get(gid_v, 'missingValue')
                v_values[v_values == v_missing_value] = np.nan
            except ecc.CodesInternalError:
                pass
            ecc.codes_release(gid_v)

        with open(gust_path, 'rb') as f:
            gid_gust = ecc.codes_grib_new_from_file(f)
            gust_values = ecc.codes_get_array(gid_gust, 'values')
            try:
                gust_missing_value = ecc.codes_get(gid_gust, 'missingValue')
                gust_values[gust_values == gust_missing_value] = np.nan
            except ecc.CodesInternalError:
                pass
            ecc.codes_release(gid_gust)

        grib_load_duration = time.perf_counter() - grib_load_start
        print(f"    ⏱️ GRIB-Ladezeit: {grib_load_duration:.4f}s")

        wind_calc_start = time.perf_counter()
        wind_pts = np.sqrt(u_values**2 + v_values**2) * 1.94384
        min_len = min(self.total_dwd_points, len(wind_pts))
        current_wind_pts = wind_pts[:min_len].astype(np.float64)

        u_slice = u_values[:min_len].astype(np.float64)
        v_slice = v_values[:min_len].astype(np.float64)
        raw_dir_deg = 270.0 - np.degrees(np.arctan2(v_slice, u_slice))
        wind_dir_pts = np.mod(raw_dir_deg, 360.0)
        current_gust_pts = gust_values[:min_len].astype(np.float64) * 1.94384

        wind_calc_duration = time.perf_counter() - wind_calc_start
        print(f"    ⏱️ Windberechnung (Knots, Richtung, Böen): {wind_calc_duration:.4f}s")

        rounded_wind_pts = np.round(current_wind_pts, 1)
        rounded_wind_dir = np.round(wind_dir_pts, 0)
        rounded_gust_pts = np.round(current_gust_pts, 0)

        # ---------------------------------------------------------------------
        # TEIL A: BILD GENERIERUNG
        # ---------------------------------------------------------------------
        interp_start = time.perf_counter()
        self.interpolator.values[:, 0] = current_wind_pts
        grid_data = self.interpolator(self.grid_x, self.grid_y)
        interp_duration = time.perf_counter() - interp_start
        print(f"    ⏱️ Interpolation: {interp_duration:.4f}s")

        color_map_start = time.perf_counter()
        conditions = [
            grid_data < 3, grid_data < 5, grid_data < 6, grid_data < 7,
            grid_data < 8, grid_data < 9, grid_data < 10, grid_data < 12,
            grid_data < 15, grid_data < 20, grid_data < 25, grid_data >= 25
        ]

        color_palette = np.array([
            [0, 0, 0, 0], [230, 255, 255, 255], [0, 191, 255, 255],
            [0, 255, 204, 255], [0, 204, 0, 255], [153, 255, 0, 255],
            [255, 255, 0, 255], [209, 158, 0, 255], [255, 85, 0, 255],
            [255, 0, 0, 255], [255, 51, 153, 255], [153, 0, 204, 255],
            [0, 0, 255, 255]
        ], dtype=np.uint8)

        choices_indices = np.arange(1, len(conditions) + 1)
        selected_color_indices = np.select(conditions, choices_indices, default=0)
        img_array = color_palette[selected_color_indices]

        color_map_duration = time.perf_counter() - color_map_start
        print(f"    ⏱️ Color Mapping (np.select): {color_map_duration:.4f}s")

        img = Image.fromarray(img_array, 'RGBA')

        # 1. PNG Speichern
        # png_save_start = time.perf_counter()
        png_filename = filename if filename.endswith(".png") else f"{filename}.png"
        # output_png_path = os.path.join(self.output_folder, png_filename)
        # img.save(output_png_path, compress_level=6)
        # png_save_duration = time.perf_counter() - png_save_start
        # print(f"    ⏱️ PNG Speichern (compress_level=6): {png_save_duration:.4f}s")

        # 2. WebP Speichern (Lossless, method=4)
        webp_save_start = time.perf_counter()
        webp_filename = filename.replace(".png", ".webp") if filename.endswith(".png") else f"{filename}.webp"
        output_webp_path = os.path.join(self.output_folder, webp_filename)
        img.save(output_webp_path, format="WEBP", lossless=True, method=4)
        webp_save_duration = time.perf_counter() - webp_save_start
        print(f"    ⏱️ WebP Speichern (Lossless, method=4): {webp_save_duration:.4f}s")

        # ---------------------------------------------------------------------
        # TEIL B: PMTILES ERSTELLUNG (0.02° WGS84 Resampling)
        # ---------------------------------------------------------------------
        pm_start = time.perf_counter()
        self.pm_interpolator_u.values[:, 0] = u_slice
        self.pm_interpolator_v.values[:, 0] = v_slice

        u_grid_002 = self.pm_interpolator_u(self.pm_grid_lon, self.pm_grid_lat).astype(np.float32)
        v_grid_002 = self.pm_interpolator_v(self.pm_grid_lon, self.pm_grid_lat).astype(np.float32)

        # Auf 2 Nachkommastellen runden (eliminiert Interpolations-Noise & steigert Gzip enorm)
        u_grid_002 = np.round(u_grid_002, decimals=2).astype(np.float32)
        v_grid_002 = np.round(v_grid_002, decimals=2).astype(np.float32)

        pmtiles_filename = filename.split('.')[0] + "_dir.pmtiles"
        output_pmtiles_path = os.path.join(self.output_folder, pmtiles_filename)
        self._create_wind_direction_pmtiles(u_grid_002, v_grid_002, output_pmtiles_path)
        print(f"    ⏱️ PMTiles (0.02°) generiert in: {time.perf_counter() - pm_start:.4f}s")

        # ---------------------------------------------------------------------
        # TEIL C: EFFIZIENTES JSON-UPDATE IM RAM (OHNE REDUNDANZEN)
        # ---------------------------------------------------------------------
        json_agg_start = time.perf_counter()
        for (col, row), meta in self.cluster_mapping.items():
            cluster_key = (col, row)

            if cluster_key not in self.cluster_memory:
                cluster_filename = os.path.join(self.cluster_output_folder, f"cluster_{col}_{row}.json")
                if os.path.exists(cluster_filename):
                    with open(cluster_filename, "rb") as jf:
                        try:
                            self.cluster_memory[cluster_key] = orjson.loads(jf.read())
                        except Exception:
                            self.cluster_memory[cluster_key] = {
                                "col": col, "row": row,
                                "lats": meta["lats"], "lons": meta["lons"],
                                "timeline": {}
                            }
                else:
                    self.cluster_memory[cluster_key] = {
                        "col": col, "row": row,
                        "lats": meta["lats"], "lons": meta["lons"],
                        "timeline": {}
                    }

            idx = meta["indices"]

            self.cluster_memory[cluster_key]["timeline"][time_key] = {
                "speeds": rounded_wind_pts[idx],
                "dirs": rounded_wind_dir[idx],
                "gusts": rounded_gust_pts[idx]
            }

            if len(self.cluster_memory[cluster_key]["timeline"]) > self.timeLineLength:
                sorted_keys = sorted(self.cluster_memory[cluster_key]["timeline"].keys())
                del self.cluster_memory[cluster_key]["timeline"][sorted_keys[0]]

        json_agg_duration = time.perf_counter() - json_agg_start
        print(f"    ⏱️ JSON RAM Aggregation (Flache Parallel-Arrays): {json_agg_duration:.4f}s")

        total_step_duration = time.perf_counter() - step_start_time
        print(f"    ⏱️ Total process_step duration: {total_step_duration:.4f}s")

        return True

    def flush_json_to_disk(self):
        flush_start_time = time.perf_counter()
        if not self.cluster_memory:
            print("💾 [Processor] Kein Cluster-Speicher zum Schreiben vorhanden.")
            return
        print(f"\n💾 [Processor] Schreibe {len(self.cluster_memory)} optimierte JSON-Cluster gesammelt in den lokalen Output-Ordner...")

        total_serialization_time = 0.0
        total_write_time = 0.0

        for (col, row), cluster_data in self.cluster_memory.items():
            cluster_filename = os.path.join(self.cluster_output_folder, f"cluster_{col}_{row}.json")

            serialization_start = time.perf_counter()
            json_bytes = orjson.dumps(cluster_data, option=orjson.OPT_SERIALIZE_NUMPY)
            serialization_duration = time.perf_counter() - serialization_start
            total_serialization_time += serialization_duration

            write_start = time.perf_counter()
            with open(cluster_filename, "wb") as json_file:
                json_file.write(json_bytes)
            write_duration = time.perf_counter() - write_start
            total_write_time += write_duration

        flush_duration = time.perf_counter() - flush_start_time
        print(f"✅ Alle JSON-Files im lokalen Output-Ordner gespeichert! (Flush time: {flush_duration:.4f}s)")
        print(f"    ⏱️ Summe JSON Serialisierung (orjson): {total_serialization_time:.4f}s")
        print(f"    ⏱️ Summe File Writing (file.write): {total_write_time:.4f}s")
