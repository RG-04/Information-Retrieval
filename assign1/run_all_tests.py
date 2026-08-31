#!/usr/bin/env python3
"""Run all test cases from test.md and compare results."""

import subprocess
import sys

# Test cases from test.md: (query_and, expected_and_count, expected_and_docids)
tests = [
    ("aeroelastic AND aircraft", 5, [12, 14, 78, 184, 202]),
    ("dynamics AND effects", 34, [110, 140, 190, 201, 210, 286, 290, 297, 328, 342, 395, 531, 650, 714, 766, 783, 792, 858, 859, 905, 939, 953, 1001, 1004, 1008, 1009, 1066, 1144, 1165, 1203, 1289, 1296, 1321, 1331]),
    ("hypersonic AND wake", 5, [17, 85, 536, 976, 1183]),
    ("flutter AND steady", 11, [14, 52, 363, 380, 444, 704, 753, 894, 899, 1272, 1339]),
    ("viscosity AND reynolds", 23, [7, 43, 50, 54, 55, 62, 73, 80, 115, 132, 151, 185, 255, 328, 417, 630, 671, 941, 962, 1007, 1082, 1159, 1226]),
    ("heat AND stagnation", 80, [24, 54, 62, 82, 84, 101, 110, 123, 142, 272, 283, 294, 303, 328, 329, 349, 352, 353, 354, 364, 366, 369, 375, 437, 438, 486, 522, 524, 539, 553, 555, 559, 564, 565, 571, 572, 576, 603, 628, 629, 635, 645, 662, 667, 668, 689, 707, 773, 876, 899, 983, 1003, 1040, 1096, 1097, 1098, 1099, 1100, 1101, 1104, 1143, 1158, 1159, 1161, 1191, 1198, 1204, 1215, 1222, 1236, 1258, 1279, 1295, 1307, 1344, 1348, 1386, 1393, 1394, 1395]),
    ("oscillatory AND transonic", 1, [815]),
    ("creep AND buckling", 26, [833, 950, 951, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1034, 1035, 1052]),
    ("pressure AND wing", 91, [14, 76, 92, 97, 147, 189, 191, 205, 222, 225, 226, 227, 230, 247, 250, 252, 287, 289, 311, 364, 379, 415, 416, 431, 464, 465, 466, 486, 513, 545, 599, 601, 612, 633, 636, 637, 652, 671, 672, 673, 675, 679, 680, 681, 692, 693, 694, 695, 696, 704, 714, 757, 781, 783, 791, 794, 798, 801, 808, 809, 811, 901, 917, 921, 924, 927, 970, 971, 991, 1062, 1064, 1074, 1075, 1090, 1091, 1092, 1111, 1144, 1184, 1188, 1207, 1208, 1229, 1233, 1239, 1246, 1271, 1276, 1277, 1336, 1355]),
    ("transonic AND nozzle", 3, [118, 157, 750]),
    ("excitation AND noise", 5, [209, 722, 725, 909, 911]),
    ("mass AND flutter", 6, [380, 442, 701, 747, 874, 1290]),
]

or_tests = [
    ("aeroelastic OR aircraft", 84),
    ("dynamics OR effects", 586),
    ("hypersonic OR wake", 211),
    ("flutter OR steady", 151),
    ("viscosity OR reynolds", 239),
    ("heat OR stagnation", 360),
    ("oscillatory OR transonic", 83),
    ("creep OR buckling", 136),
    ("pressure OR wing", 687),
    ("transonic OR nozzle", 140),
    ("excitation OR noise", 47),
    ("mass OR flutter", 121),
]

print("=" * 80)
print("RUNNING AND TESTS")
print("=" * 80)

and_results = []
for i, (query, expected_count, expected_docs) in enumerate(tests, 1):
    cmd = ["python", "QuadQuery_pa1.py", "search", "QuadQuery_cran.index", "stopwords.txt", query]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse the output
    output_lines = result.stdout.strip().split('\n')
    result_str = output_lines[-1]  # Last line should be the result
    actual_docs = eval(result_str)  # Convert string representation to list
    
    matches = actual_docs == expected_docs
    status = "✓ PASS" if matches else "✗ FAIL"
    
    print(f"\nTest {i}: {query}")
    print(f"  Expected count: {expected_count}, Actual count: {len(actual_docs)} {status}")
    
    if not matches:
        print(f"  Expected: {expected_docs}")
        print(f"  Got:      {actual_docs}")
        
        # Find mismatches
        missing = set(expected_docs) - set(actual_docs)
        extra = set(actual_docs) - set(expected_docs)
        if missing:
            print(f"  Missing: {sorted(missing)}")
        if extra:
            print(f"  Extra: {sorted(extra)}")
    
    and_results.append((query, matches, len(actual_docs), expected_count))

print("\n" + "=" * 80)
print("RUNNING OR TESTS")
print("=" * 80)

or_results = []
for i, (query, expected_count) in enumerate(or_tests, 1):
    cmd = ["python", "QuadQuery_pa1.py", "search", "QuadQuery_cran.index", "stopwords.txt", query]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse the output
    output_lines = result.stdout.strip().split('\n')
    result_str = output_lines[-1]  # Last line should be the result
    actual_docs = eval(result_str)  # Convert string representation to list
    
    matches = len(actual_docs) == expected_count
    status = "✓ PASS" if matches else "✗ FAIL"
    
    print(f"\nTest {i}: {query}")
    print(f"  Expected count: {expected_count}, Actual count: {len(actual_docs)} {status}")
    
    or_results.append((query, matches, len(actual_docs), expected_count))

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

and_passed = sum(1 for _, passed, _, _ in and_results if passed)
or_passed = sum(1 for _, passed, _, _ in or_results if passed)

print(f"\nAND Tests: {and_passed}/{len(and_results)} passed")
print(f"OR Tests:  {or_passed}/{len(or_results)} passed")
print(f"Total:     {and_passed + or_passed}/{len(and_results) + len(or_results)} passed")

if and_passed == len(and_results) and or_passed == len(or_results):
    print("\n✓ ALL TESTS PASSED!")
    sys.exit(0)
else:
    print("\n✗ SOME TESTS FAILED")
    sys.exit(1)
