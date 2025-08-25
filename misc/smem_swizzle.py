# Parameters
ELEM_BYTES = 2
NUM_ROWS = 64
NUM_COLS = 64

# Calculations
tile_row_dim = 16
tile_col_dim = 32 if ELEM_BYTES == 1 else 16
underlying_height = NUM_ROWS // tile_row_dim
underlying_width = NUM_COLS // tile_col_dim

# Decide swizzle bytes
if ELEM_BYTES == 1:
    if underlying_width % 4 == 0:
        swizzle_subtile_col_bytes = 128
    elif underlying_width % 2 == 0:
        swizzle_subtile_col_bytes = 64
    else:
        swizzle_subtile_col_bytes = 32
elif ELEM_BYTES == 2:
    if underlying_width % 4 == 0:
        swizzle_subtile_col_bytes = 128
    elif underlying_width % 2 == 0:
        swizzle_subtile_col_bytes = 64
    else:
        swizzle_subtile_col_bytes = 32
elif ELEM_BYTES == 4:
    if underlying_width % 2 == 0:
        swizzle_subtile_col_bytes = 128
    else:
        swizzle_subtile_col_bytes = 64

# Asserts
assert ELEM_BYTES in [1, 2, 4]
assert NUM_ROWS % tile_row_dim == 0
assert NUM_COLS % tile_col_dim == 0
assert (ELEM_BYTES * NUM_COLS) % swizzle_subtile_col_bytes == 0

# Original tile is divided into multiple "swizzle subtiles"
swizzle_subtile_rows = NUM_ROWS
swizzle_subtile_cols = swizzle_subtile_col_bytes // ELEM_BYTES
swizzle_subtile_nelem = swizzle_subtile_cols * swizzle_subtile_rows
swizzle_subtile_bytes = swizzle_subtile_nelem * ELEM_BYTES

# Within a swizzle subtile, swizzling pattern is repeated every 8 rows
swizzle_subtile_pattern_repeat_nelem = swizzle_subtile_cols * 8
swizzle_subtile_pattern_repeat_bytes = swizzle_subtile_pattern_repeat_nelem * ELEM_BYTES

# Not really needed, but helper values to print out
swizzle_subsubtile_per_subtile = swizzle_subtile_nelem // swizzle_subtile_pattern_repeat_nelem
swizzle_128B_groups_per_subsubtile = swizzle_subtile_pattern_repeat_bytes // 128

# Print metadata
print("ELEM_BYTES:", ELEM_BYTES)
print("NUM_ROWS:", NUM_ROWS)
print("NUM_COLS:", NUM_COLS)
print("tile_row_dim:", tile_row_dim)
print("tile_col_dim:", tile_col_dim)
print("underlying_height:", underlying_height)
print("underlying_width:", underlying_width)
print("swizzle_subtile_col_bytes:", swizzle_subtile_col_bytes)
print("swizzle_subtile_rows:", swizzle_subtile_rows)
print("swizzle_subtile_cols:", swizzle_subtile_cols)
print("swizzle_subtile_bytes:", swizzle_subtile_bytes)
print("swizzle_subtile_pattern_repeat_nelem:", swizzle_subtile_pattern_repeat_nelem)
print("swizzle_subtile_pattern_repeat_bytes:", swizzle_subtile_pattern_repeat_bytes)
print("swizzle_subsubtile_per_subtile:", swizzle_subsubtile_per_subtile)
print("swizzle_128B_groups_per_subsubtile:", swizzle_128B_groups_per_subsubtile)

# Naive (r, c) --> swizzled index
def swizzled_index(row_idx: int, col_idx: int) -> int:
    # Dividing original tile into multiple swizzle subtiles,
    # what is the index of this swizzle subtile?
    swizzle_subtile_idx = col_idx // swizzle_subtile_cols # only divide by cols, since subtile takes entire rows
    within_swizzle_subtile_row_idx = row_idx # since subtile takes entire row
    within_swizzle_subtile_col_idx = col_idx % swizzle_subtile_cols

    # Now that we know swizzle subtile outer idx and inner row & col, we can calculate idx in memory
    idx_in_memory = swizzle_subtile_idx * swizzle_subtile_nelem + \
                    within_swizzle_subtile_row_idx * swizzle_subtile_cols + \
                    within_swizzle_subtile_col_idx % swizzle_subtile_cols
    
    # From here, it's theoretical. Assume base address is 0, calculate the address in memory
    base_addr = 0
    addr_in_memory = base_addr + idx_in_memory * ELEM_BYTES

    # Within swizzle subtile, the pattern is repeated every 8 rows
    # Let's call this swizzle sub-subtile
    # Find out the byte offset in swizzle sub-subtile
    byte_offset_in_swizzle_subsubtile = addr_in_memory % swizzle_subtile_pattern_repeat_bytes
    
    # The swizzle sub-subtile is further divided into groups of 128B. Let's get the index of this
    idx_of_128B_group_in_swizzle_subsubtile = byte_offset_in_swizzle_subsubtile >> 7

    # Finally, swizzle!
    # Group 0 does nothing
    # Group 1 flips bit 4
    # Group 2 flips bit 5
    # Group 3 flips bits 4 and 5
    # Group 4 flips bit 6
    # Group 5 flips bits 4 and 6
    # Group 6 flips bits 5 and 6
    # Group 7 flips bits 4, 5, and 6
    swizzled_addr_in_memory = addr_in_memory ^ (idx_of_128B_group_in_swizzle_subsubtile << 4)

    return swizzled_addr_in_memory

# Print out the layout
swizzled_layout = [[None for _ in range(NUM_COLS)] for _ in range(NUM_ROWS)]
for row_idx in range(NUM_ROWS):
    for col_idx in range(NUM_COLS):
        swizzled_addr_in_memory = swizzled_index(row_idx, col_idx)
        swizzled_idx = swizzled_addr_in_memory // ELEM_BYTES
        swizzled_layout_row_idx = swizzled_idx // NUM_COLS
        swizzled_layout_col_idx = swizzled_idx % NUM_COLS
        swizzled_layout[swizzled_layout_row_idx][swizzled_layout_col_idx] = (row_idx, col_idx)

# Print out the layout
for row_idx in range(NUM_ROWS):
    print(f"Row {row_idx:3d}", end=" | ")
    for col_idx in range(NUM_COLS):
        row_idx, col_idx = swizzled_layout[row_idx][col_idx]
        print(f"({row_idx:3d}, {col_idx:3d})", end=" ")
    print(f"\n        |")
