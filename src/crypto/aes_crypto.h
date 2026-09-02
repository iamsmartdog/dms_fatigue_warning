#ifndef DMS_AES_CRYPTO_H
#define DMS_AES_CRYPTO_H

#include <cstdint>
#include <array>
#include <mutex>
#include <vector>

namespace dms {

// AES-256-CTR 加密模块（单例）
//
// 设计动机（隐私合规）：
// - 存证视频必须落盘加密，原始码流不出设备
// - 用单例统一管理密钥，避免多处持有；CTR 模式无填充、可流式加密
// - 与封装解耦：Muxer 在写入每个 sample 前调用 encrypt() 处理
//
// 安全说明：本实现仅做"传输/落盘"层加密，密钥派生自设备唯一标识，
// 防止视频文件被直接拷贝播放；如需更高强度需接入硬件密钥库（OP-TEE）
class AesCrypto {
public:
    static constexpr size_t kKeyBytes   = 32;  // AES-256
    static constexpr size_t kNonceBytes = 12;  // CTR 通常 96-bit nonce

    // 单例访问
    static AesCrypto& instance();

    // 初始化：传入 32B 密钥（或由设备 ID 派生）
    // 重复调用会重置计数器，慎用
    void init(const uint8_t key[kKeyBytes]);

    // 是否已就绪
    bool ready() const { return ready_; }

    // 流式加密：in -> out（长度不变），CTR 计数器自动累加
    // 线程安全（内部加锁，避免多 muxer 同时写）
    void encrypt(const uint8_t* in, size_t n, std::vector<uint8_t>& out);

    // 便捷重载
    void encrypt(const std::vector<uint8_t>& in, std::vector<uint8_t>& out) {
        encrypt(in.data(), in.size(), out);
    }

    // 由设备 ID（CPU序列号/MAC 等）派生密钥（KDF：简化版 SHA-256 折叠）
    static void derive_key_from_device_id(const std::string& device_id,
                                          uint8_t key[kKeyBytes]);

    AesCrypto(const AesCrypto&) = delete;
    AesCrypto& operator=(const AesCrypto&) = delete;

private:
    AesCrypto() = default;
    ~AesCrypto() = default;

    // AES-CTR 核心：用 counter 加密 keystream，再与明文异或
    void ctr_update(const uint8_t* in, uint8_t* out, size_t n);

    std::array<uint8_t, kKeyBytes>   key_{};
    std::array<uint8_t, kNonceBytes> nonce_{};
    uint64_t counter_ = 0;        // 块计数器
    size_t   block_offset_ = 0;   // 当前 keystream 块内偏移（0-15），跨调用保持
    uint8_t  keystream_[16] = {}; // 当前块的 keystream 缓存
    bool     ready_   = false;
    std::mutex mtx_;
};

}  // namespace dms

#endif  // DMS_AES_CRYPTO_H
