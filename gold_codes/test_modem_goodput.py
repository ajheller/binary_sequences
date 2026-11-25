import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
import numpy as np
import mary_gold_ssb_modem_v1 as m

class GoodputTests(unittest.TestCase):
    def test_failed_and_wrong_packets_cost_time(self):
        payloads=[]
        def transmit(payload,*args):
            payloads.append(payload)
            return np.zeros([8000,16000,8000][len(payloads)-1]),0.,0.
        def receive(*args,**kwargs):
            i=len(payloads)
            if i==1:return m.DecodeResult(payloads[-1],'ok')
            if i==2:return m.DecodeResult(None,'crc_failed')
            return m.DecodeResult(bytes(16),'ok') # Even accepted wrong data is not goodput.
        out=io.StringIO()
        with patch.object(m,'TRIALS_PER_SNR',3),patch.object(m,'SNR_SWEEP_DB',[-12]), \
             patch.object(m,'modulate_packet',side_effect=transmit), \
             patch.object(m,'pass_through_ssb_channel',side_effect=lambda audio,*args:audio), \
             patch.object(m,'demodulate_packet',side_effect=receive),redirect_stdout(out):
            m.run_ber_sweep(np.ones(127),np.ones((16,127)),np.ones(1))
        row=next(line for line in out.getvalue().splitlines() if '-12.0 dB' in line)
        cells=[s.strip() for s in row.split('|')]
        self.assertEqual(cells[2],'1/3')
        # Three 128-bit attempts over 1+2+1 seconds; only the first delivered.
        self.assertEqual(float(cells[-2]),96.)
        self.assertEqual(float(cells[-1]),32.)

    def test_empty_and_no_attempts(self):
        self.assertEqual(m.payload_rates(0,0,2),(0,0))
        self.assertTrue(all(np.isnan(x) for x in m.payload_rates(0,0,0)))

    def test_code_length_and_pilot_overhead(self):
        h=m.rrc_filter(m.SPS,m.RRC_SPAN,m.ROLLOFF)
        rates=[]
        for degree in (7,9):
            p,c=m.generate_gold_codes(degree)
            for config in (m.PilotConfig(),m.PilotConfig('periodic',interval=4)):
                audio,*_=m.modulate_packet(bytes(16),p,c,h,pilot_config=config)
                rates.append(m.payload_rates(128,128,len(audio)/m.FS)[0])
        self.assertGreater(rates[0],rates[1])
        self.assertGreater(rates[0],rates[2])
        self.assertGreater(rates[2],rates[3])
        self.assertAlmostEqual(rates[0],128/5.9025)

if __name__=='__main__':unittest.main(verbosity=2)
