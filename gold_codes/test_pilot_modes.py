import unittest
from unittest.mock import patch
import numpy as np
from scipy import signal
import mary_gold_ssb_modem_v1 as m

class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h=m.rrc_filter(m.SPS,m.RRC_SPAN,m.ROLLOFF)

    def test_reserved_sequences_and_packet_boundaries(self):
        for degree in (7,9):
            pre,data=m.generate_gold_codes(degree)
            pilot=m.payload_pilot_code(pre)
            self.assertFalse(np.array_equal(pre,pilot))
            for reserved in (pre,pilot):
                self.assertFalse(any(np.array_equal(reserved,row) for row in data))
                ac=np.rint(np.fft.ifft(abs(np.fft.fft(reserved))**2).real)
                np.testing.assert_array_equal(ac[1:],-np.ones(len(pre)-1))
            np.testing.assert_array_equal(data[0],pre*np.roll(pilot,15))
            for i in range(1,16):
                np.testing.assert_array_equal(data[i],pre*np.roll(pilot,i-1))
            for cfg in (m.PilotConfig('periodic',interval=4),m.PilotConfig('continuous')):
                audio,_,_=m.modulate_packet(b'Boundary',pre,data,self.h,pilot_config=cfg)
                np.testing.assert_array_equal(audio[:2000],np.zeros(2000))
                np.testing.assert_array_equal(audio[-2000:],np.zeros(2000))
        with self.assertRaises(ValueError):m.generate_gold_codes(7,128)
        with self.assertRaises(ValueError):m.payload_pilot_code(-pre)

    def test_config_header_integrity(self):
        for config in (m.PilotConfig(),m.PilotConfig('periodic',interval=4,burst_length=2),
                       m.PilotConfig('continuous',power_fraction=.05),m.PilotConfig('continuous',power_fraction=.5)):
            frame=m.encode_frame(b'123',config)
            self.assertEqual(m.validate_frame(frame),(b'123','ok'))
            parsed=m.parse_header(frame[:m.HEADER_BYTES])
            self.assertEqual(parsed[0],3)
            self.assertEqual(parsed[1].wire_bytes(),config.wire_bytes())
            for k in range(m.HEADER_BYTES*8):
                bad=bytearray(frame[:m.HEADER_BYTES]);bad[k//8]^=1<<(k%8)
                self.assertIsNone(m.parse_header(bad))
        for kwargs in ({'mode':'bad'},{'power_fraction':0},{'power_fraction':.501},
                       {'power_fraction':float('nan')},{'power_fraction':.123},
                       {'interval':0},{'interval':256},{'burst_length':0},{'burst_length':17}):
            with self.assertRaises(ValueError):m.PilotConfig(**kwargs)

    def test_schedule_and_power(self):
        pre,data=m.generate_gold_codes(7)
        cfg=m.PilotConfig('periodic',interval=4,burst_length=2)
        self.assertEqual(m.body_schedule(9,cfg),[0,1,2,3,None,None,4,5,6,7,None,None,8])
        self.assertEqual(m.body_schedule(4,cfg),[0,1,2,3])
        for power in (.05,.1,.2,.5):
            cfg=m.PilotConfig('continuous',power_fraction=power)
            templates,raw=m.body_templates(pre,data,cfg)
            np.testing.assert_allclose(np.mean(templates**2,axis=1),1,atol=1e-12)
            expected=1+2*np.sqrt(power*(1-power))*np.mean(m.payload_pilot_code(pre)*data,axis=1)
            np.testing.assert_allclose(raw,expected)
            tx=m.modulate_packet(b'Hi',pre,data,self.h,pilot_config=cfg,return_details=True)
            self.assertEqual(tx.pilot_symbols,0)
            self.assertGreater(tx.active_audio_power,0)
            self.assertLessEqual(np.max(abs(tx.audio)),.80000001)

    def test_mode_loopbacks(self):
        configs=[m.PilotConfig(),m.PilotConfig('periodic',interval=4,burst_length=2),
                 m.PilotConfig('continuous',power_fraction=.05),m.PilotConfig('continuous',power_fraction=.2),
                 m.PilotConfig('continuous',power_fraction=.5)]
        for n in (7,9):
            pre,data=m.generate_gold_codes(n)
            for cfg in configs:
                for payload in (b'',b'Pilots'):
                    with self.subTest(n=n,cfg=cfg,payload=payload):
                        audio,_,_=m.modulate_packet(payload,pre,data,self.h,pilot_config=cfg)
                        # Local TX settings deliberately disagree; RX must read header.
                        with patch.object(m,'PILOT_MODE','continuous'):
                            out=m.demodulate_packet(audio,pre,data,self.h,return_details=True)
                        self.assertEqual(out.payload,payload,out)
                        self.assertEqual(out.pilot_config.wire_bytes(),cfg.wire_bytes())
                        if cfg.mode=='continuous':self.assertGreater(out.pilot_observations,0)

    def test_awgn_and_tracking_switch(self):
        pre,data=m.generate_gold_codes(7)
        for cfg in (m.PilotConfig('periodic',interval=4),m.PilotConfig('continuous',power_fraction=.2)):
            audio,_,_=m.modulate_packet(b'Test',pre,data,self.h,pilot_config=cfg)
            for offset in (-38.5,38.5):
                rx=m.pass_through_ssb_channel(audio,-6,offset,73,np.random.default_rng(321))
                for enabled in (False,True):
                    out=m.demodulate_packet(rx,pre,data,self.h,return_details=True,pilot_tracking=enabled)
                    self.assertEqual(out.payload,b'Test',out)
                    if not enabled:self.assertEqual(out.tracking_updates,0)

    def test_frequency_drift(self):
        pre,data=m.generate_gold_codes(7)
        for cfg in (m.PilotConfig('periodic',interval=4),m.PilotConfig('continuous',power_fraction=.2)):
            payload=b'Frequency drift!'
            tx=m.modulate_packet(payload,pre,data,self.h,pilot_config=cfg,return_details=True)
            t=np.arange(len(tx.audio))/m.FS
            body_time=.25+(m.PREAMBLE_LEN+m.HEADER_BYTES*2)*len(pre)/m.CHIP_RATE
            phase=2*np.pi*(20*t+5*np.maximum(t-body_time,0)**2) # +10 Hz/s after header
            rx=np.real(signal.hilbert(tx.audio)*np.exp(1j*phase))
            enabled=m.demodulate_packet(rx,pre,data,self.h,return_details=True,pilot_tracking=True)
            disabled=m.demodulate_packet(rx,pre,data,self.h,return_details=True,pilot_tracking=False)
            print('\nDrift:',cfg.mode,enabled.status,enabled.tracking_updates,enabled.final_vfo_hz,disabled.status,flush=True)
            self.assertEqual(enabled.payload,payload,enabled)
            self.assertGreater(enabled.final_vfo_hz,35)
            self.assertIsNone(disabled.payload)

    def test_sample_timing_drift(self):
        pre,data=m.generate_gold_codes(7)
        payload=b'Sample clock drift'
        for cfg in (m.PilotConfig('periodic',interval=4),m.PilotConfig('continuous',power_fraction=.2)):
            audio,_,_=m.modulate_packet(payload,pre,data,self.h,pilot_config=cfg)
            start=2000+(m.PREAMBLE_LEN+m.HEADER_BYTES*2)*len(pre)*m.SPS
            grid=np.arange(len(audio))
            delay=10*np.clip((grid-start)/(len(audio)-2000-start),0,1)
            rx=np.interp(grid-delay,grid,audio)
            out=m.demodulate_packet(rx,pre,data,self.h,return_details=True)
            print('\nTiming drift:',cfg.mode,out.status,out.timing_offset_samples,flush=True)
            self.assertEqual(out.payload,payload,out)
            self.assertGreaterEqual(out.timing_offset_samples,7)

    def test_false_preamble_and_truncation(self):
        pre,data=m.generate_gold_codes(7)
        cfg=m.PilotConfig('continuous',power_fraction=.5)
        audio,_,_=m.modulate_packet(b'Frame start',pre,data,self.h,pilot_config=cfg)
        # Leading high-power pilot-only block has a higher correlation score.
        chips=np.tile(pre,8)
        up=np.zeros(len(chips)*m.SPS);up[::m.SPS]=chips
        bb=np.convolve(up,self.h)
        fake=bb*np.cos(2*np.pi*m.F_SUB*np.arange(len(bb))/m.FS)
        prefix=np.r_[np.zeros(2000),fake,np.zeros(2000)]
        out=m.demodulate_packet(np.r_[prefix,audio],pre,data,self.h,return_details=True)
        self.assertEqual(out.payload,b'Frame start',out)
        self.assertGreater(out.sync_idx,len(prefix))
        rx=audio[:-(2000+3*len(pre)*m.SPS)]
        self.assertIsNone(m.demodulate_packet(rx,pre,data,self.h)[0])

if __name__=='__main__':unittest.main(verbosity=2)
