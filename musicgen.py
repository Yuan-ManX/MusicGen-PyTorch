import typing as tp
import warnings

import torch

from encodec import CompressionModel
from genmodel import BaseGenModel
from lm import LMModel
from builders import get_debug_compression_model, get_debug_lm_model
from loaders import load_compression_model, load_lm_model
from audio_utils import convert_audio
from conditioners import ConditioningAttributes, WavCondition, StyleConditioner


# 定义 MelodyList 类型，表示一个包含可选张量的列表
MelodyList = tp.List[tp.Optional[torch.Tensor]]
# 定义 MelodyType 类型，表示张量或 MelodyList 类型
MelodyType = tp.Union[torch.Tensor, MelodyList]


# backward compatible names mapping
# 定义与 Hugging Face 模型检查点名称的映射，提供向后兼容的名称
_HF_MODEL_CHECKPOINTS_MAP = {
    "small": "facebook/musicgen-small",
    "medium": "facebook/musicgen-medium",
    "large": "facebook/musicgen-large",
    "melody": "facebook/musicgen-melody",
    "style": "facebook/musicgen-style",
}


class MusicGen(BaseGenModel):
    """MusicGen main model with convenient generation API.

    Args:
        name (str): name of the model.
        compression_model (CompressionModel): Compression model
            used to map audio to invertible discrete representations.
        lm (LMModel): Language model over discrete representations.
        max_duration (float, optional): maximum duration the model can produce,
            otherwise, inferred from the training params.
    """
    """
    MusicGen 主模型，提供便捷的音乐生成 API。

    参数:
        name (str): 模型的名称。
        compression_model (CompressionModel): 压缩模型，用于将音频映射到可量化的离散表示。
        lm (LMModel): 语言模型，用于处理离散表示。
        max_duration (float, 可选): 模型可以生成的最大时长。如果未提供，则根据训练参数推断。
    """
    def __init__(self, name: str, compression_model: CompressionModel, lm: LMModel,
                 max_duration: tp.Optional[float] = None):
        super().__init__(name, compression_model, lm, max_duration)
        self.set_generation_params(duration=15)  # default duration

    @staticmethod
    def get_pretrained(name: str = 'facebook/musicgen-melody', device=None):
        """Return pretrained model, we provide four models:
        - facebook/musicgen-small (300M), text to music,
          # see: https://huggingface.co/facebook/musicgen-small
        - facebook/musicgen-medium (1.5B), text to music,
          # see: https://huggingface.co/facebook/musicgen-medium
        - facebook/musicgen-melody (1.5B) text to music and text+melody to music,
          # see: https://huggingface.co/facebook/musicgen-melody
        - facebook/musicgen-large (3.3B), text to music,
          # see: https://huggingface.co/facebook/musicgen-large
        - facebook/musicgen-style (1.5 B), text and style to music,
          # see: https://huggingface.co/facebook/musicgen-style
        """
        """
        参数:
            name (str, 可选): 预训练模型的名称，默认为 'facebook/musicgen-melody'。
            device (Optional[str], 可选): 模型加载的设备，'cuda' 或 'cpu'。如果未提供，则自动选择。

        返回:
            MusicGen: 预训练的 MusicGen 模型实例。
        """
        if device is None:
            if torch.cuda.device_count():
                device = 'cuda'
            else:
                device = 'cpu'

        if name == 'debug':
            # used only for unit tests
            # 仅用于单元测试
            # 获取调试压缩模型
            compression_model = get_debug_compression_model(device)
            # 获取调试语言模型
            lm = get_debug_lm_model(device)
            # 返回调试模型实例
            return MusicGen(name, compression_model, lm, max_duration=30)

        if name in _HF_MODEL_CHECKPOINTS_MAP:
            # 如果使用旧版检查点名称映射，则发出警告
            warnings.warn(
                "MusicGen pretrained model relying on deprecated checkpoint mapping. " +
                f"Please use full pre-trained id instead: facebook/musicgen-{name}")
            # 使用映射后的名称
            name = _HF_MODEL_CHECKPOINTS_MAP[name]

        # 加载语言模型和压缩模型
        lm = load_lm_model(name, device=device)
        compression_model = load_compression_model(name, device=device)
        if 'self_wav' in lm.condition_provider.conditioners:
            # 如果语言模型包含 'self_wav' 条件提供者，则设置匹配长度和掩码使用
            lm.condition_provider.conditioners['self_wav'].match_len_on_eval = True
            lm.condition_provider.conditioners['self_wav']._use_masking = False

        return MusicGen(name, compression_model, lm)

    def set_generation_params(self, use_sampling: bool = True, top_k: int = 250,
                              top_p: float = 0.0, temperature: float = 1.0,
                              duration: float = 30.0, cfg_coef: float = 3.0,
                              cfg_coef_beta: tp.Optional[float] = None,
                              two_step_cfg: bool = False, extend_stride: float = 18,):
        """Set the generation parameters for MusicGen.

        Args:
            use_sampling (bool, optional): Use sampling if True, else do argmax decoding. Defaults to True.
            top_k (int, optional): top_k used for sampling. Defaults to 250.
            top_p (float, optional): top_p used for sampling, when set to 0 top_k is used. Defaults to 0.0.
            temperature (float, optional): Softmax temperature parameter. Defaults to 1.0.
            duration (float, optional): Duration of the generated waveform. Defaults to 30.0.
            cfg_coef (float, optional): Coefficient used for classifier free guidance. Defaults to 3.0.
            cfg_coef_beta (float, optional): beta coefficient in double classifier free guidance.
                Should be only used for MusicGen melody if we want to push the text condition more than
                the audio conditioning. See paragraph 4.3 in https://arxiv.org/pdf/2407.12563 to understand
                double CFG.
            two_step_cfg (bool, optional): If True, performs 2 forward for Classifier Free Guidance,
                instead of batching together the two. This has some impact on how things
                are padded but seems to have little impact in practice.
            extend_stride: when doing extended generation (i.e. more than 30 seconds), by how much
                should we extend the audio each time. Larger values will mean less context is
                preserved, and shorter value will require extra computations.
        """
        """
        设置 MusicGen 的生成参数。

        参数:
            use_sampling (bool, 可选): 是否使用采样（True），否则使用 argmax 解码。默认为 True。
            top_k (int, 可选): 采样时使用的 top_k 参数。默认为 250。
            top_p (float, 可选): 采样时使用的 top_p 参数，当设置为 0 时使用 top_k。默认为 0.0。
            temperature (float, 可选): Softmax 温度参数。默认为 1.0。
            duration (float, 可选): 生成音频的时长。默认为 30.0 秒。
            cfg_coef (float, 可选): 分类器引导的系数。默认为 3.0。
            cfg_coef_beta (float, 可选): 双重分类器引导中的 beta 系数。
                仅在 MusicGen melody 中使用，用于增加文本条件的权重。
                参见文档第 4.3 节以了解双重 CFG。
            two_step_cfg (bool, 可选): 是否执行两步分类器引导（True），而不是将两者一起批处理。
                这会影响填充方式，但在实践中影响不大。
            extend_stride (float): 当进行扩展生成（即超过 30 秒）时，每次扩展音频的步长。
                较大的值意味着保留的上下文较少，较小的值则需要额外的计算。
        """
        # 确保扩展步长不超过最大生成时长
        assert extend_stride < self.max_duration, "Cannot stride by more than max generation duration."
        self.extend_stride = extend_stride
        self.duration = duration
        self.generation_params = {
            'use_sampling': use_sampling,
            'temp': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'cfg_coef': cfg_coef,
            'two_step_cfg': two_step_cfg,
            'cfg_coef_beta': cfg_coef_beta,
        }

    def set_style_conditioner_params(self, eval_q: int = 3, excerpt_length: float = 3.0,
                                     ds_factor: tp.Optional[int] = None,
                                     encodec_n_q: tp.Optional[int] = None) -> None:
        """Set the parameters of the style conditioner
        Args:
            eval_q (int): the number of residual quantization streams used to quantize the style condition
                the smaller it is, the narrower is the information bottleneck
            excerpt_length (float): the excerpt length in seconds that is extracted from the audio
                conditioning
            ds_factor: (int): the downsampling factor used to downsample the style tokens before
                using them as a prefix
            encodec_n_q: (int, optional): if encodec is used as a feature extractor, sets the number
                of streams that is used to extract features
        """
        """
        设置风格条件器的参数。

        参数:
            eval_q (int): 用于量化风格条件的残差量化流的数量。
                数量越小，信息瓶颈越窄。
            excerpt_length (float): 从音频条件中提取的片段长度（秒）。
            ds_factor (int, 可选): 用于在将风格标记用作前缀之前进行下采样的下采样因子。
            encodec_n_q (int, 可选): 如果使用 Encodec 作为特征提取器，则设置用于提取特征的流数量。
        """
        assert isinstance(self.lm.condition_provider.conditioners.self_wav, StyleConditioner), \
            "Only use this function if you model is MusicGen-Style"
        self.lm.condition_provider.conditioners.self_wav.set_params(eval_q=eval_q,
                                                                    excerpt_length=excerpt_length,
                                                                    ds_factor=ds_factor,
                                                                    encodec_n_q=encodec_n_q)

    def generate_with_chroma(self, descriptions: tp.List[str], melody_wavs: MelodyType,
                             melody_sample_rate: int, progress: bool = False,
                             return_tokens: bool = False) -> tp.Union[torch.Tensor,
                                                                      tp.Tuple[torch.Tensor, torch.Tensor]]:
        """Generate samples conditioned on text and melody.

        Args:
            descriptions (list of str): A list of strings used as text conditioning.
            melody_wavs: (torch.Tensor or list of Tensor): A batch of waveforms used as
                melody conditioning. Should have shape [B, C, T] with B matching the description length,
                C=1 or 2. It can be [C, T] if there is a single description. It can also be
                a list of [C, T] tensors.
            melody_sample_rate: (int): Sample rate of the melody waveforms.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        """
        根据文本和旋律生成样本。

        参数:
            descriptions (List[str]): 用于文本条件生成的字符串列表。
            melody_wavs (torch.Tensor 或 List[torch.Tensor]): 用于旋律条件生成的音频波形批次。
                形状应为 [B, C, T]，其中 B 与描述长度匹配，C=1 或 2。
                如果只有一个描述，则可以是 [C, T]。
                也可以是 [C, T] 张量的列表。
            melody_sample_rate (int): 旋律波形的采样率。
            progress (bool, 可选): 是否显示生成过程的进度。默认为 False。

        返回:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: 返回生成的音频张量。
                如果 return_tokens 为 True，则返回生成的音频和相应的 tokens。
        """
        if isinstance(melody_wavs, torch.Tensor):
            if melody_wavs.dim() == 2:
                # 如果旋律波形的维度为 2，则添加批次维度
                melody_wavs = melody_wavs[None]
            if melody_wavs.dim() != 3:
                # 确保旋律波形的形状为 [B, C, T]
                raise ValueError("Melody wavs should have a shape [B, C, T].")
            # 将旋律波形转换为列表
            melody_wavs = list(melody_wavs)
        else:
            for melody in melody_wavs:
                if melody is not None:
                    assert melody.dim() == 2, "One melody in the list has the wrong number of dims."
        # 将旋律波形转换为目标采样率和通道数
        melody_wavs = [
            convert_audio(wav, melody_sample_rate, self.sample_rate, self.audio_channels)
            if wav is not None else None
            for wav in melody_wavs]
        # 准备 tokens 和属性
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions=descriptions, prompt=None,
                                                                        melody_wavs=melody_wavs)
        assert prompt_tokens is None
        # 生成 tokens
        tokens = self._generate_tokens(attributes, prompt_tokens, progress)
        if return_tokens:
            return self.generate_audio(tokens), tokens
        return self.generate_audio(tokens)

    @torch.no_grad()
    def _prepare_tokens_and_attributes(
            self,
            descriptions: tp.Sequence[tp.Optional[str]],
            prompt: tp.Optional[torch.Tensor],
            melody_wavs: tp.Optional[MelodyList] = None,
    ) -> tp.Tuple[tp.List[ConditioningAttributes], tp.Optional[torch.Tensor]]:
        """Prepare model inputs.

        Args:
            descriptions (list of str): A list of strings used as text conditioning.
            prompt (torch.Tensor): A batch of waveforms used for continuation.
            melody_wavs (torch.Tensor, optional): A batch of waveforms
                used as melody conditioning. Defaults to None.
        """
        """
        准备模型输入。

        参数:
            descriptions (List[Optional[str]]): 用于文本条件生成的字符串列表。
            prompt (torch.Tensor): 用于继续生成音频的音频波形批次。
            melody_wavs (List[Optional[torch.Tensor]], 可选): 用于旋律条件生成的音频波形批次。默认为 None。

        返回:
            Tuple[List[ConditioningAttributes], Optional[torch.Tensor]]: 返回属性列表和提示 tokens。
        """
        attributes = [
            ConditioningAttributes(text={'description': description}) # 创建属性对象
            for description in descriptions] # 为每个描述创建一个属性对象

        if melody_wavs is None:
            # 如果没有旋律波形，则为每个属性添加一个空的旋律条件
            for attr in attributes:
                attr.wav['self_wav'] = WavCondition(
                    torch.zeros((1, 1, 1), device=self.device),
                    torch.tensor([0], device=self.device),
                    sample_rate=[self.sample_rate],
                    path=[None])
        else:
            # 如果存在旋律波形，则检查模型是否支持旋律条件
            if 'self_wav' not in self.lm.condition_provider.conditioners:
                raise RuntimeError("This model doesn't support melody conditioning. "
                                   "Use the `melody` model.")
            assert len(melody_wavs) == len(descriptions), \
                f"number of melody wavs must match number of descriptions! " \
                f"got melody len={len(melody_wavs)}, and descriptions len={len(descriptions)}"
            for attr, melody in zip(attributes, melody_wavs):
                if melody is None:
                    # 如果旋律波形为 None，则添加一个空的旋律条件
                    attr.wav['self_wav'] = WavCondition(
                        # 创建一个零张量
                        torch.zeros((1, 1, 1), device=self.device),
                        # 创建长度张量
                        torch.tensor([0], device=self.device),
                        # 设置采样率
                        sample_rate=[self.sample_rate],
                        path=[None])
                else:
                    # 否则，将旋律波形添加到属性中
                    attr.wav['self_wav'] = WavCondition(
                        melody[None].to(device=self.device),
                        # 创建长度张量
                        torch.tensor([melody.shape[-1]], device=self.device),
                        # 设置采样率
                        sample_rate=[self.sample_rate],
                        path=[None],
                    )

        if prompt is not None:
            # 如果存在提示张量，则检查描述数量与提示数量是否匹配
            if descriptions is not None:
                assert len(descriptions) == len(prompt), "Prompt and nb. descriptions doesn't match"
            prompt = prompt.to(self.device)
            prompt_tokens, scale = self.compression_model.encode(prompt)
            assert scale is None
        else:
            prompt_tokens = None
        return attributes, prompt_tokens

    def _generate_tokens(self, attributes: tp.List[ConditioningAttributes],
                         prompt_tokens: tp.Optional[torch.Tensor], progress: bool = False) -> torch.Tensor:
        """Generate discrete audio tokens given audio prompt and/or conditions.

        Args:
            attributes (list of ConditioningAttributes): Conditions used for generation (text/melody).
            prompt_tokens (torch.Tensor, optional): Audio prompt used for continuation.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        Returns:
            torch.Tensor: Generated audio, of shape [B, C, T], T is defined by the generation params.
        """
        """
        根据音频提示和/或条件生成离散音频 tokens。

        参数:
            attributes (List[ConditioningAttributes]): 用于生成的条件（文本/旋律）。
            prompt_tokens (Optional[torch.Tensor]): 用于继续生成的音频提示。
            progress (bool, 可选): 是否显示生成过程的进度。默认为 False。

        返回:
            torch.Tensor: 生成的音频 tokens，形状为 [B, C, T]，T 由生成参数定义。
        """
        # 计算总生成长度
        total_gen_len = int(self.duration * self.frame_rate)
        # 计算最大提示长度
        max_prompt_len = int(min(self.duration, self.max_duration) * self.frame_rate)
        # 当前生成偏移量
        current_gen_offset: int = 0

        def _progress_callback(generated_tokens: int, tokens_to_generate: int):
            # 更新生成的 tokens
            generated_tokens += current_gen_offset
            if self._progress_callback is not None:
                # Note that total_gen_len might be quite wrong depending on the
                # codebook pattern used, but with delay it is almost accurate.
                self._progress_callback(generated_tokens, tokens_to_generate)
            else:
                print(f'{generated_tokens: 6d} / {tokens_to_generate: 6d}', end='\r')

        if prompt_tokens is not None:
            # 回调函数，用于显示进度
            assert max_prompt_len >= prompt_tokens.shape[-1], \
                "Prompt is longer than audio to generate"

        callback = None
        if progress:
            callback = _progress_callback

        if self.duration <= self.max_duration:
            # generate by sampling from LM, simple case.
            with self.autocast:
                gen_tokens = self.lm.generate(
                    prompt_tokens, attributes,
                    callback=callback, max_gen_len=total_gen_len, **self.generation_params)

        else:
            # now this gets a bit messier, we need to handle prompts,
            # melody conditioning etc.
            ref_wavs = [attr.wav['self_wav'] for attr in attributes]
            all_tokens = []
            if prompt_tokens is None:
                prompt_length = 0
            else:
                all_tokens.append(prompt_tokens)
                prompt_length = prompt_tokens.shape[-1]

            assert self.extend_stride is not None, "Stride should be defined to generate beyond max_duration"
            assert self.extend_stride < self.max_duration, "Cannot stride by more than max generation duration."
            stride_tokens = int(self.frame_rate * self.extend_stride)

            while current_gen_offset + prompt_length < total_gen_len:
                time_offset = current_gen_offset / self.frame_rate
                chunk_duration = min(self.duration - time_offset, self.max_duration)
                max_gen_len = int(chunk_duration * self.frame_rate)
                for attr, ref_wav in zip(attributes, ref_wavs):
                    wav_length = ref_wav.length.item()
                    if wav_length == 0:
                        continue
                    # We will extend the wav periodically if it not long enough.
                    # we have to do it here rather than in conditioners.py as otherwise
                    # we wouldn't have the full wav.
                    initial_position = int(time_offset * self.sample_rate)
                    wav_target_length = int(self.max_duration * self.sample_rate)
                    positions = torch.arange(initial_position,
                                             initial_position + wav_target_length, device=self.device)
                    attr.wav['self_wav'] = WavCondition(
                        ref_wav[0][..., positions % wav_length],
                        torch.full_like(ref_wav[1], wav_target_length),
                        [self.sample_rate] * ref_wav[0].size(0),
                        [None], [0.])
                with self.autocast:
                    gen_tokens = self.lm.generate(
                        prompt_tokens, attributes,
                        callback=callback, max_gen_len=max_gen_len, **self.generation_params)
                if prompt_tokens is None:
                    all_tokens.append(gen_tokens)
                else:
                    all_tokens.append(gen_tokens[:, :, prompt_tokens.shape[-1]:])
                prompt_tokens = gen_tokens[:, :, stride_tokens:]
                prompt_length = prompt_tokens.shape[-1]
                current_gen_offset += stride_tokens

            gen_tokens = torch.cat(all_tokens, dim=-1)
        return gen_tokens
