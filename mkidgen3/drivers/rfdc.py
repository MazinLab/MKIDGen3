def status():
    import mkidgen3
    rfdc = mkidgen3._gen3_overlay.usp_rf_data_converter_0

    regmap = {'Restart Power-On State Machine': 0x0004,
              'Restart State': 0x0008,
              'Current State': 0x000C,
              'Reset Count': 0x0038,
              'Interrupt Status': 0x0200,
              'Tile Common Status': 0x0228,
              'Tile Disable': 0x0230}
    tilemap = [(f'ADC{i}', v) for i, v in enumerate((0x14000, 0x18000, 0x1C000, 0x20000))]
    tilemap += [(f'DAC{i}', v) for i, v in enumerate((0x04000, 0x08000))]  # , 0x0C000, 0x10000))]
    tilemap = dict(tilemap)
    print(rfdc.read(0x0008))
    for t, taddr in tilemap.items():
        print(t)
        for k, r in regmap.items():
            print(f'  {k}:  {rfdc.read(taddr + r)}')


def reset():
    import mkidgen3
    rfdc = mkidgen3._gen3_overlay.usp_rf_data_converter_0
    rfdc.write(0x0004, 0x00000001)
