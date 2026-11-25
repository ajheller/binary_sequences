import unittest
import tempfile
from pathlib import Path
import numpy as np
import mary_gold_ssb_modem_v1 as m

class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = m.rrc_filter(m.SPS, m.RRC_SPAN, m.ROLLOFF)

    def test_code_families(self):
        for n in (7, 9):
            pre, data = m.generate_gold_codes(n)
            codes = np.vstack([pre, data])
            self.assertEqual(codes.shape, (17, 2**n-1))
            bound = 1 + 2**((n+1)//2)
            for i in range(len(codes)):
                for j in range(i):
                    cc = np.fft.ifft(np.fft.fft(codes[i])*np.conj(np.fft.fft(codes[j]))).real
                    self.assertLessEqual(np.max(np.abs(cc)), bound+1e-8)
        for n, count in [(8,16), (7,0), (7,129)]:
            with self.assertRaises(ValueError):
                m.generate_gold_codes(n,count)

    def test_crc_and_lengths(self):
        self.assertEqual(m.zlib.crc32(b'123456789'), 0xcbf43926)
        for payload in (b'', b'abc', bytes(range(255))):
            frame=m.encode_frame(payload)
            self.assertEqual(m.validate_frame(frame), (payload,'ok'))
            for k in range(len(frame)*8):
                bad=bytearray(frame); bad[k//8] ^= 1 << (k%8)
                self.assertIsNone(m.validate_frame(bad)[0])
            self.assertIsNone(m.validate_frame(frame[:-1])[0])
            self.assertIsNone(m.validate_frame(frame+b'\0')[0])
        with self.assertRaises(ValueError): m.encode_frame(bytes(256))

    def test_filter_tail(self):
        pre,data=m.generate_gold_codes(7)
        old=m.ENABLE_CFR; m.ENABLE_CFR=False
        try:
            audio,_,_=m.modulate_packet(b'A',pre,data,self.h)
        finally: m.ENABLE_CFR=old
        frame=np.frombuffer(m.encode_frame(b'A'),dtype=np.uint8)
        indices=np.column_stack([frame>>4,frame&15]).ravel()
        chips=np.concatenate([np.tile(pre,m.PREAMBLE_LEN),data[indices].ravel()])
        up=np.zeros(len(chips)*m.SPS); up[::m.SPS]=chips
        bb=np.convolve(up,self.h,'full')
        expected=bb*np.cos(2*np.pi*m.F_SUB*np.arange(len(bb))/m.FS)
        expected=expected/max(abs(expected))*0.8
        np.testing.assert_allclose(audio[2000:-2000],expected,atol=5e-12)
        self.assertTrue(np.any(abs(expected[len(up):])>1e-6))

    def test_loopbacks(self):
        for n in (7,9):
            pre,data=m.generate_gold_codes(n)
            for payload in (b'', b'Hi'):
                audio,_,_=m.modulate_packet(payload,pre,data,self.h)
                result=m.demodulate_packet(audio,pre,data,self.h,return_details=True)
                self.assertEqual(result.payload,payload,result)
                self.assertTrue(result.ok)
            audio,_,_=m.modulate_packet(b'Hi',pre,data,self.h)
            for offset in (-38.5,38.5):
                rx=m.pass_through_ssb_channel(audio,-6,offset,73,np.random.default_rng(123))
                result=m.demodulate_packet(rx,pre,data,self.h,return_details=True)
                self.assertEqual(result.payload,b'Hi',result)
                self.assertLess(abs(result.vfo_hz-offset),2)

    def test_maximum_payload(self):
        pre,data=m.generate_gold_codes(7)
        payload=bytes(range(255))
        audio,_,_=m.modulate_packet(payload,pre,data,self.h)
        self.assertEqual(m.demodulate_packet(audio,pre,data,self.h)[0],payload)

    def test_rejection(self):
        pre,data=m.generate_gold_codes(7)
        audio,_,_=m.modulate_packet(b'Hi',pre,data,self.h)
        for rx in (np.zeros(0), np.zeros_like(audio),
                   *[np.random.default_rng(seed).normal(size=len(audio)) for seed in range(5)],
                   np.cos(2*np.pi*1500*np.arange(len(audio))/m.FS),
                   audio[:-(2000+3*len(pre)*m.SPS)]):
            result=m.demodulate_packet(rx,pre,data,self.h,return_details=True)
            self.assertIsNone(result.payload,result)
            self.assertFalse(result.ok)
        # Deliberately valid modulation of a frame with a damaged CRC.
        original=m.encode_frame
        def damaged(payload, pilot_config=None):
            frame=original(payload, pilot_config)
            return frame[:-1]+bytes([frame[-1]^1])
        m.encode_frame=damaged
        try: bad,_,_=m.modulate_packet(b'Hi',pre,data,self.h)
        finally: m.encode_frame=original
        result=m.demodulate_packet(bad,pre,data,self.h,return_details=True)
        self.assertEqual(result.status,'crc_failed',result)
        self.assertEqual(result.raw_payload,b'Hi')

    def test_wav(self):
        with tempfile.TemporaryDirectory() as d:
            p=str(Path(d)/'audio.wav')
            x=np.array([-8.25,0.,7.75])
            m.save_wav(p,x,float_wav=True)
            np.testing.assert_array_equal(m.load_wav(p),x)
            m.save_wav(p,x)
            np.testing.assert_allclose(m.load_wav(p),np.clip(x,-1,1),atol=1/32768)

if __name__=='__main__': unittest.main(verbosity=2)
