import numpy as np
import sounddevice as sd
from scipy.signal import butter, filtfilt
import time

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
        self.buffer = np.array([], dtype=np.float32)
        self.demodulated_bits = []  # буфер для демодулированных битов
        self.current_mode = 'LAT'  # текущий режим декодирования

    def _bandpass_filter(self, signal, low, high, order=5):
        """Полосовой фильтр для выделения частоты."""
        nyq = 0.5 * self.sample_rate
        low = low / nyq
        high = high / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal)


    def _detect_frequency(self, segment):
        # Уменьшаем окно до 80% бита для точности
        window = int(len(segment) * 0.8)
        segment = segment[-window:]  # берём конец сегмента
        
        mark_filtered = self._bandpass_filter(segment, self.mark_freq - 60, self.mark_freq + 60)
        space_filtered = self._bandpass_filter(segment, self.space_freq - 60, self.space_freq + 60)
        
        mark_energy = np.sum(mark_filtered ** 2)
        space_energy = np.sum(space_filtered ** 2)
        
        
        # Гистерезис: не переключаться, если разница < 10%
        if abs(mark_energy - space_energy) < 0.2 * (mark_energy + space_energy):
            return 'mark' if mark_energy > space_energy else 'space'
        
        return 'mark' if mark_energy > space_energy else 'space'


    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback‑функция для потокового захвата аудио."""
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32768.0
        self.buffer = np.concatenate((self.buffer, audio_data))


        while len(self.buffer) >= self.n_samples_per_bit:
            # Берём один сегмент (один бит)
            segment = self.buffer[:self.n_samples_per_bit]
            self.buffer = self.buffer[self.n_samples_per_bit:]


            # Демодуляция сегмента в бит
            freq_type = self._detect_frequency(segment)
            bit = 1 if freq_type == 'mark' else 0

            # Сохраняем демодулированный бит
            self.demodulated_bits.append(bit)


        if len(self.buffer) > 10 * self.n_samples_per_bit:
            self.buffer = self.buffer[-10 * self.n_samples_per_bit:]
        return in_data, 'continue'  # Исправленный возврат

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
        current_mode = self.current_mode  # берём текущий режим из атрибута
        i = 0  # индекс в массиве bits

        while i < len(bits) - 6:  # нужно минимум 7 битов (старт + 5 данных + стоп)
            # Проверяем старт‑бит (должен быть 0)
            if bits[i] != 0:
                i += 1
                continue

            # Извлекаем 5 битов данных
            data_bits = bits[i + 1: i + 6]

            # Ищем стоп‑бит: минимум одна 1 после данных
            j = i + 6
            stop_length = 0
            while j < len(bits) and bits[j] == 1:
                stop_length += 1
                j += 1

            # Если стоп‑бит не найден (нет ни одной 1), пропускаем
            if stop_length < 1:
                i += 1
                continue

            # Декодируем символ
            char = self._decode_ita2_char(data_bits, current_mode)

            # Обрабатываем переключение режимов
            if char == 'RUS':
                current_mode = 'RUS'
                self.current_mode = 'RUS'  # ← сохраняем в атрибут
            elif char == 'FIGS':
                current_mode = 'FIGS'
                self.current_mode = 'FIGS'  # ← сохраняем в атрибут
            elif char == 'LAT':
                current_mode = 'LAT'
                self.current_mode = 'LAT'
            else:
                text.append(char)
    
            # Переходим к следующему возможному старт‑биту (после стоп‑бита)
            i = j

        return ''.join(text), i



    def _process_decoding(self):
        """Декодирует накопленные биты и выводит текст."""
        if len(self.demodulated_bits) < 7:
            return  # мало битов — ждём

        # Декодируем
        text, num_bits = self.decode_bits(self.demodulated_bits)
        if text:
            text = text.replace('\r', '\n')  # упрощённая обработка \r
            while '\n\n' in text:
                text = text.replace('\n\n', '\n')
            print(text, end='', flush=True)
            # Удаляем только обработанные биты
            self.demodulated_bits = self.demodulated_bits[num_bits:]

    def start_streaming(self):
        """Запускает потоковую обработку с микрофона."""
        self.demodulated_bits = []
        self.current_mode = 'LAT'

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='int16',
                callback=self.audio_callback
            ):
                print("Начата потоковая обработка с микрофона. Для остановки нажмите Ctrl+C.")
                while True:
                    time.sleep(0.1)  # пауза для CPU
                    self._process_decoding()
        except KeyboardInterrupt:
            print("\nПотоковая обработка остановлена.")
        except Exception as e:
            print(f"Ошибка: {e}")



# Пример использования
if __name__ == '__main__':
    
    decoder = RTTYDecoder(
        baud=45.45,           # должно совпадать с передатчиком
        mark_freq=1170,       # например, 1170 Гц
        space_freq=1000,      # например, 1000 Гц
        sample_rate=44100      # обычно 44100 или 48000
    )
    decoder.start_streaming()

