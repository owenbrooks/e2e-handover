global_force_torques = ['gfX', 'gfY', 'gfZ', 'gtX', 'gtY', 'gtZ']
pillar_disp_force = ['dX', 'dY', 'dZ', 'fX', 'fY', 'fZ'] 
papillarray_keys = []
for sensor in [1, 2]:
    for key in global_force_torques:
        papillarray_keys.append(f'tact_{sensor}_{key}')

    for i in range(9):
        for key in pillar_disp_force:
            papillarray_keys.append(f'tact_{sensor}_pil_{i}_{key}')

def sensor_state_to_list(msg):
    values = [msg.gfX, msg.gfY, msg.gfZ, msg.gtX, msg.gtY, msg.gtZ]

    for pil in msg.pillars: # will be 9 pillars
        values += [pil.dX, pil.dY, pil.dZ, pil.fX, pil.fY, pil.fZ]

    return values
