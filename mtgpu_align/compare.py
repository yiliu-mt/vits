import os
import numpy as np

input_dir = 'mtgpu_align'
filenames = set()

for filename in os.listdir('mtgpu_align/cpu'):
    filenames.add(filename.split('_')[0])

for filename in filenames:
    cpu_enc_x = np.load(f'mtgpu_align/cpu/{filename}_enc_x.npy')
    cpu_enc_m_p = np.load(f'mtgpu_align/cpu/{filename}_enc_m_p.npy')
    cpu_enc_logs_p = np.load(f'mtgpu_align/cpu/{filename}_enc_logs_p.npy')
    cpu_g = np.load(f'mtgpu_align/cpu/{filename}_g.npy')
    cpu_dp_logw = np.load(f'mtgpu_align/cpu/{filename}_dp_logw.npy')
    cpu_dp_w = np.load(f'mtgpu_align/cpu/{filename}_dp_w.npy')
    cpu_dp_attn = np.load(f'mtgpu_align/cpu/{filename}_dp_attn.npy')
    cpu_flow_m_p = np.load(f'mtgpu_align/cpu/{filename}_flow_m_p.npy')
    cpu_flow_logs_p = np.load(f'mtgpu_align/cpu/{filename}_flow_logs_p.npy')
    cpu_flow_z_p = np.load(f'mtgpu_align/cpu/{filename}_flow_z_p.npy')
    cpu_flow_z = np.load(f'mtgpu_align/cpu/{filename}_flow_z.npy')
    cpu_output = np.load(f'mtgpu_align/cpu/{filename}_output.npy')

    mt_enc_x = np.load(f'mtgpu_align/mtgpu/{filename}_enc_x.npy')
    mt_enc_m_p = np.load(f'mtgpu_align/mtgpu/{filename}_enc_m_p.npy')
    mt_enc_logs_p = np.load(f'mtgpu_align/mtgpu/{filename}_enc_logs_p.npy')
    mt_g = np.load(f'mtgpu_align/mtgpu/{filename}_g.npy')
    mt_dp_logw = np.load(f'mtgpu_align/mtgpu/{filename}_dp_logw.npy')
    mt_dp_w = np.load(f'mtgpu_align/mtgpu/{filename}_dp_w.npy')
    mt_dp_attn = np.load(f'mtgpu_align/mtgpu/{filename}_dp_attn.npy')
    mt_flow_m_p = np.load(f'mtgpu_align/mtgpu/{filename}_flow_m_p.npy')
    mt_flow_logs_p = np.load(f'mtgpu_align/mtgpu/{filename}_flow_logs_p.npy')
    mt_flow_z_p = np.load(f'mtgpu_align/mtgpu/{filename}_flow_z_p.npy')
    mt_flow_z = np.load(f'mtgpu_align/mtgpu/{filename}_flow_z.npy')
    mt_output = np.load(f'mtgpu_align/mtgpu/{filename}_output.npy')

    np.allclose(cpu_enc_x, mt_enc_x, rtol=1e-4, atol=1e-4)
    np.allclose(cpu_enc_m_p, mt_enc_m_p, rtol=1e-4, atol=1e-4)
    np.allclose(cpu_enc_logs_p, mt_enc_logs_p, rtol=1e-4, atol=1e-4)
    np.allclose(cpu_g, mt_g, rtol=1e-4, atol=1e-4)
    np.allclose(cpu_dp_logw, mt_dp_logw, rtol=1e-4, atol=1e-4)
    np.allclose(cpu_dp_w, mt_dp_w, rtol=1e-4, atol=1e-4)
    np.allclose(cpu_dp_attn, mt_dp_attn, rtol=1e-4, atol=1e-4)
    np.allclose(cpu_flow_m_p, mt_flow_m_p, rtol=1e-4, atol=1e-4)
    np.allclose(cpu_flow_logs_p, mt_flow_logs_p, rtol=1e-4, atol=1e-4)
    np.allclose(cpu_flow_z_p, mt_flow_z_p, rtol=1e-4, atol=1e-4)
    np.allclose(cpu_flow_z, mt_flow_z, rtol=1e-4, atol=1e-4)
    np.allclose(cpu_output, mt_output, rtol=1e-4, atol=1e-4)
