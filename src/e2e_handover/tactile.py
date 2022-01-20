global_force_torques = ['tact_gfX', 'tact_gfY', 'tact_gfZ', 'tact_gtX', 'tact_gtY', 'tact_gtZ']
pillar_disp_force = ['dX', 'dY', 'dZ', 'fX', 'fY', 'fZ'] 
pillar_keys = []
for i in range(9):
    for key in pillar_disp_force:
        pillar_keys.append(f'tact_{i}_{key}')
papillarray_keys = global_force_torques + pillar_keys

def sensor_state_to_list(msg):
    values = [msg.gfX, msg.gfY, msg.gfZ, msg.gtX, msg.gtY, msg.gtZ]

    for pil in msg.pillars: # will be 9 pillars
        values += [pil.dX, pil.dY, pil.dZ, pil.fX, pil.fY, pil.fZ]

    return values
