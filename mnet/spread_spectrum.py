# simple spread-spectrum with NumPy

from dataclasses import dataclass
import numpy as np


# https://docs.python.org/3/library/dataclasses.html
@dataclass
class SpreadSpectrum:
    number: int = 8
    length: int = 256
    seed: int = 0

    def spreading_matrix(self):
        m = np.random.default_rng(seed=self.seed).choice(
            [-1, 1], (self.number, self.length)
        )
        return m.astype(np.float32)

    def binary_to_bipolar(self, m):
        return 2 * m.astype(np.int32) - 1

    def bipolar_to_bytes(self, bips):
        bytes = bytearray(np.packbits(bips > 0))
        return bytes

    def bytes_to_bipolar(self, data):
        bips = self.binary_to_bipolar(np.unpackbits(data).reshape(-1, self.number))
        return bips

    def modulate(self, data):
        sm = self.spreading_matrix()
        # print("sm", sm.shape)
        bips = self.binary_to_bipolar(np.unpackbits(data).reshape(-1, self.number))
        # print(bits)
        tx_signal = np.array([b @ sm for b in bips])
        return tx_signal

    def demodulate(self, rx_signal):
        sm = self.spreading_matrix()
        bipolar_bits = np.array([sm @ s for s in rx_signal])
        return bipolar_bits


def unit_test(tx_data=b"abcdefgh", n_codes=8, code_length=256):
    ss = SpreadSpectrum(length=code_length, number=number_of_codes)
    print(ss)
    tx_signal = ss.modulate(bytearray(tx_data))
    weights_per_byte = tx_signal.size / len(tx_data)
    print("weights/byte", weights_per_byte)
    # print(np.std(tx))

    # channel
    rx_signal = tx_signal

    rx = ss.demodulate(rx_signal)
    # print(rx)
    print("rx", rx.shape, rx.size)
    integration_gain = np.std(rx) / np.std(tx_signal)
    print(
        "integration gain",
        integration_gain,
        20 * np.log10(integration_gain),
        "dB",
    )
    rx_data = ss.bipolar_to_bytes(rx)

    print("err", np.array(rx_data).shape, np.array(tx_data).shape)
    ber = np.mean(rx_data != tx_data)
    return ber, weights_per_byte, integration_gain


if __name__ == "__main__":
    # print(modulate(np.uint8(bytearray(b"abc"))))

    # print(spreading_matrix().shape)
    # tx = modulate(bytearray(b"abcd"))
    # rx = demodulate(tx)
    # data = bipolar_to_bytes(rx)
    # print(data)
    N_CODES = 8
    _data = np.random.randint(0, 255, size=1 << 12, dtype=np.uint32)
    with open("test.csv", "w") as f:
        print("length,", "n_codes,", "weights/byte,", "ber,", "gain,", "dB,", file=f)
        for code_length in [1024]:  # [32, 64, 128, 256, 512, 1024, 2048]:
            for number_of_codes in [8]:  # [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
                ber, weights_per_byte, integration_gain = unit_test(
                    _data, n_codes=number_of_codes, code_length=code_length
                )
                r = f"{code_length}, {number_of_codes}, {weights_per_byte}, {ber}, {integration_gain}, {20 * np.log10(integration_gain)}"
                print(r)
