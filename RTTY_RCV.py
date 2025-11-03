import numpy as np
from scipy.signal import find_peaks
from scipy.io import wavfile
import matplotlib.pyplot as plt

class RTTYDecoder:
    """Декодирует RTTY‑сигнал (FSK) в текст по стандарту ITA2."""
    # Специальные коды переключения режимов
    MODE_SWITCH = {
        (0, 0, 0, 0, 0): 'RUS',
        (1, 1, 0, 1, 1): 'FIGS',
        (1, 1, 1, 1, 1): 'LAT',
    }
    ITA2_MODES = {
        'LAT': {
            (1, 1, 0, 0, 0): 'A', (1, 0, 0, 1, 1): 'B', (0, 1, 1, 1, 0): 'C',
            (1, 0, 0, 1, 0): 'D', (1, 0, 0, 0, 0): 'E', (1, 0, 1, 1, 0): 'F',
            (0, 1, 0, 1, 1): 'G', (0, 0, 1, 0, 1): 'H', (0, 1, 1, 0, 0): 'I',
            (1, 1, 0, 1, 0): 'J', (1, 1, 1, 1, 0): 'K', (0, 1, 0, 0, 1): 'L',
            (0, 0, 1, 1, 1): 'M', (0, 0, 1, 1, 0): 'N', (0, 0, 0, 1, 1): 'O',
            (0, 1, 1, 0, 1): 'P', (1, 1, 1, 0, 1): 'Q', (0, 1, 0, 1, 0): 'R',
            (1, 0, 1, 0, 0): 'S', (0, 0, 0, 0, 1): 'T', (1, 1, 1, 0, 0): 'U',
            (0, 1, 1, 1, 1): 'V', (1, 1, 0, 0, 1): 'W', (1, 0, 1, 1, 1): 'X',
            (1, 0, 1, 0, 1): 'Y', (1, 0, 0, 0, 1): 'Z', (0, 0, 1, 0, 0): ' ',
            (0, 0, 0, 1, 0): '\r', (0, 1, 0, 0, 0): '\n',
        },
        'RUS': {
            (1, 1, 0, 0, 0): 'А', (1, 0, 0, 1, 1): 'Б', (1, 1, 0, 0, 1): 'В',
            (0, 1, 0, 1, 1): 'Г', (1, 0, 0, 1, 0): 'Д', (1, 0, 0, 0, 0): 'Е',
            (0, 1, 1, 1, 1): 'Ж', (1, 0, 0, 0, 1): 'З', (0, 1, 1, 0, 0): 'И',
            (1, 1, 0, 1, 0): 'Й', (1, 1, 1, 1, 0): 'К', (0, 1, 0, 0, 1): 'Л',
            (0, 0, 1, 1, 1): 'М', (0, 0, 1, 1, 0): 'Н', (0, 0, 0, 1, 1): 'О',
            (0, 1, 1, 0, 1): 'П', (0, 1, 0, 1, 0): 'Р', (1, 0, 1, 0, 0): 'С',
            (0, 0, 0, 0, 1): 'Т', (1, 1, 1, 0, 0): 'У', (1, 0, 1, 1, 0): 'Ф',
            (0, 0, 1, 0, 1): 'Х', (0, 1, 1, 1, 0): 'Ц', (1, 0, 1, 1, 1): 'Ъ',
            (1, 0, 1, 0, 1): 'Ы', (1, 0, 1, 1, 1): 'Ь', (1, 1, 1, 0, 1): 'Я',
            (1, 0, 0, 0, 0): 'Ё',
            (0, 0, 0, 1, 0): '\r', (0, 1, 0, 0, 0): '\n',
        },
        'FIGS': {
            (0, 1, 1, 0, 1): '0', (1, 1, 1, 0, 1): '1', (1, 1, 0, 0, 1): '2',
            (1, 0, 0, 0, 0): '3', (0, 1, 0, 1, 0): '4', (0, 0, 0, 0, 1): '5',
            (1, 0, 1, 0, 1): '6', (1, 1, 1, 0, 0): '7', (0, 1, 1, 0, 0): '8',
            (0, 0, 0, 1, 1): '9', (1, 1, 0, 0, 0): '-', (1, 0, 0, 0, 1): '+',
            (1, 0, 0, 1, 1): '?', (0, 1, 1, 1, 0): ':', (1, 1, 1, 1, 0): '(',
            (0, 1, 0, 0, 1): ')', (0, 0, 1, 1, 1): '.', (0, 0, 1, 1, 0): ',',
            (0, 1, 1, 1, 1): '/', (0, 0, 1, 0, 0): ' ',
            (0, 1, 0, 1, 1): 'Ш', (0, 0, 1, 0, 1): 'Щ', (1, 0, 1, 1, 0): 'Э',
            (1, 1, 0, 1, 0): 'Ю', (0, 1, 0, 1, 0): 'Ч',
            (0, 0, 0, 1, 0): '\r', (0, 1, 0, 0, 0): '\n',
        }
    }




    def __init__(self, baud=45.45, mark_freq=1170, space_freq=1000, sample_rate=44100):
        self.baud = baud
        self.mark_freq = mark_freq
        self.space_freq = space_freq
        self.sample_rate = sample_rate
        self.bit_duration = 1.0 / baud  # длительность одного бита (сек)
        self.n_samples_per_bit = int(sample_rate * self.bit_duration)


    def _bandpass_filter(self, signal, low, high, order=5):
        """Полосовой фильтр для выделения частоты."""
        from scipy.signal import butter, filtfilt
        nyq = 0.5 * self.sample_rate
        low = low / nyq
        high = high / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal)


    def _detect_frequency(self, segment):
        """Определяет, какая частота преобладает в сегменте (mark или space)."""
        # Фильтруем сигнал в диапазонах mark и space
        mark_filtered = self._bandpass_filter(segment, self.mark_freq - 50, self.mark_freq + 50)
        space_filtered = self._bandpass_filter(segment, self.space_freq - 50, self.space_freq + 50)

        # Вычисляем энергию в каждом диапазоне
        mark_energy = np.sum(mark_filtered ** 2)
        space_energy = np.sum(space_filtered ** 2)

        # Сравниваем энергии: где больше — та частота и преобладает
        return 'mark' if mark_energy > space_energy else 'space'


    def demodulate(self, signal):
        """Демодулирует RTTY‑сигнал в битовую последовательность."""
        bits = []
        n_samples = len(signal)

        for i in range(0, n_samples, self.n_samples_per_bit):
            segment = signal[i:i + self.n_samples_per_bit]
            if len(segment) < self.n_samples_per_bit // 2:
                continue  # слишком короткий сегмент — пропускаем


            # Определяем частоту в сегменте
            freq_type = self._detect_frequency(segment)
            bit = 1 if freq_type == 'mark' else 0
            bits.append(bit)

        return bits


    def _decode_ita2_char(self, code, current_mode):
        """Декодирует 5‑битный код в символ с учётом текущего режима."""
        code_tuple = tuple(code)
        if code_tuple in self.MODE_SWITCH:
            return self.MODE_SWITCH[code_tuple]
        if (current_mode in self.ITA2_MODES
                and code_tuple in self.ITA2_MODES[current_mode]):
            return self.ITA2_MODES[current_mode][code_tuple]
        return '?'


    def decode_bits(self, bits):
        """
        Декодирует битовую последовательность в текст.
        Учитывает старт/стоп‑биты и переключение режимов (LAT/RUS/FIGS).
        Поддерживает стоп‑бит длиной 1.5.
        """
        text = []
        current_mode = 'LAT'  # начальный режим
        bit_buffer = []  # буфер для битов символа
        i = 0  # индекс в массиве bits

        while i < len(bits):
            bit = bits[i]
            bit_buffer.append(bit)

            # Ищем старт‑бит (0) в начале буфера
            if len(bit_buffer) == 1 and bit != 0:
                # Не старт‑бит — сбрасываем буфер
                bit_buffer = []
                i += 1
                continue

            # Если набрали 7 битов (старт + 5 данных + стоп)
            if len(bit_buffer) >= 7:
                # Проверяем, что старт‑бит = 0
                if bit_buffer[0] != 0:
                    bit_buffer = []
                    continue

                # Извлекаем 5 битов данных
                data_bits = bit_buffer[1:6]

                # Ищем стоп‑бит: он может быть 1 (1×) или 1,1 (1.5×)
                stop_pattern = []
                j = i  # начинаем с текущего индекса
                while j < len(bits) and bits[j] == 1:
                    stop_pattern.append(bits[j])
                    j += 1

                stop_length = len(stop_pattern)

                # Если стоп‑бит есть (хотя бы 1 единица), считаем символ завершённым
                if stop_length >= 1:
                    # Декодируем символ с учётом текущего режима
                    char = self._decode_ita2_char(data_bits, current_mode)
                    
                    #print(f"Биты: {bit_buffer} → Данные: {data_bits} → Символ: {repr(char)} (режим: {current_mode})")
                    
                    if char == 'RUS':
                        current_mode = 'RUS'
                    elif char == 'FIGS':
                        current_mode = 'FIGS'
                    elif char == 'LAT':
                        current_mode = 'LAT'
                    else:
                        text.append(char)
                        # Принудительный переход в LAT после CR или LF
                        #if char in ('\r', '\n'):
                        #    current_mode = 'LAT'
                            # print(f"Принудительно переключено в LAT после символа {char}")

                    # Сдвигаем индекс на длину стоп‑бита (пропускаем его)
                    i = j
                    bit_buffer = []
                else:
                    # Нет стоп‑бита — продолжаем накапливать
                    i += 1
            else:
                i += 1

        return ''.join(text)


    def decode(self, signal_or_path):
        """
        Основной метод: принимает сигнал (массив NumPy) или путь к WAV‑файлу,
        демодулирует и декодирует в текст.
        """
        # Если передан путь к файлу — загружаем
        if isinstance(signal_or_path, str):
            sample_rate, signal = wavfile.read(signal_or_path)
            if sample_rate != self.sample_rate:
                print(f"Предупреждение: частота дискретизации {sample_rate} Гц не совпадает с ожидаемой {self.sample_rate} Гц.")
        else:
            signal = signal_or_path

        # Приводим к моно (если стерео)
        if signal.ndim > 1:
            signal = signal[:, 0]  # берём левый канал

        # Нормализуем сигнал
        signal = signal.astype(np.float32)
        signal /= np.max(np.abs(signal))


        # 1. Демодуляция: сигнал → биты
        bits = self.demodulate(signal)

        # 2. Декодирование: биты → текст
        text = self.decode_bits(bits)

        return text

if __name__ == "__main__":
    # Создаём декодер с параметрами, соответствующими передатчику
    decoder = RTTYDecoder(
        baud=45.45,
        mark_freq=1170,
        space_freq=1000,
        sample_rate=44100
    )
    
    # Вариант 1: декодирование из WAV‑файла
    file_path = "audio/rtty_message.wav"  # укажите путь к вашему файлу
    try:
        text = decoder.decode(file_path)
        print("Распознанный текст:")
        print(text)
    except FileNotFoundError:
        print(f"Файл не найден: {file_path}")
    except Exception as e:
        print(f"Ошибка при декодировании: {e}")

    # Вариант 2: декодирование из массива NumPy (если сигнал уже загружен)
    # sample_rate, signal = wavfile.read("audio/rtty_message.wav")
    # text = decoder.decode(signal)
    # print("Распознанный текст:")
    # print(text)

