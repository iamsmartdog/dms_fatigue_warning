#include "crypto/aes_crypto.h"

#include <cstring>

#ifdef DMS_HAS_OPENSSL
#include <openssl/aes.h>
#include <openssl/sha.h>
#include <openssl/rand.h>
#endif

#include "utils/log.h"

namespace dms {

AesCrypto& AesCrypto::instance() {
    static AesCrypto inst;
    return inst;
}

void AesCrypto::init(const uint8_t key[kKeyBytes]) {
    std::lock_guard<std::mutex> lock(mtx_);
    std::memcpy(key_.data(), key, kKeyBytes);
    // nonce 固定派生（密钥前 12B），实战可每文件随机 + 文件头存储
    std::memcpy(nonce_.data(), key, kNonceBytes);
    counter_ = 0;
    block_offset_ = 0;   // 重置 keystream 位置，防止密钥流重用
    ready_ = true;
    LOGI("AES", "AES-256-CTR initialized");
}

void AesCrypto::derive_key_from_device_id(const std::string& device_id,
                                          uint8_t key[kKeyBytes]) {
#ifdef DMS_HAS_OPENSSL
    uint8_t digest[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(device_id.data()),
           device_id.size(), digest);
    std::memcpy(key, digest, kKeyBytes);
#else
    // FNV-1a 64-bit 折叠到 32B（仅占位 KDF，真实环境务必用 SHA-256）
    uint64_t h = 0xcbf29ce484222325ULL;
    for (unsigned char c : device_id) {
        h ^= c;
        h *= 0x100000001b3ULL;
    }
    for (int i = 0; i < 4; ++i) {
        uint64_t v = h + i * 0x9e3779b97f4a7c15ULL;
        std::memcpy(key + i * 8, &v, 8);
    }
#endif
}

#ifdef DMS_HAS_OPENSSL

void AesCrypto::ctr_update(const uint8_t* in, uint8_t* out, size_t n) {
    // CTR：构造 16B 计数器块 = nonce(12B) + counter(4B big-endian)
    // 修复密钥流重用：block_offset_ 跨调用保持，确保每个字节用唯一 keystream
    AES_KEY aes_key;
    AES_set_encrypt_key(key_.data(), 256, &aes_key);

    uint8_t counter_blk[16];
    std::memcpy(counter_blk, nonce_.data(), kNonceBytes);

    size_t offset = 0;
    while (offset < n) {
        // 块用完（block_offset_==0）时生成新 keystream
        if (block_offset_ == 0) {
            uint32_t c = static_cast<uint32_t>(counter_);
            counter_blk[12] = (c >> 24) & 0xFF;
            counter_blk[13] = (c >> 16) & 0xFF;
            counter_blk[14] = (c >> 8) & 0xFF;
            counter_blk[15] = c & 0xFF;
            AES_encrypt(counter_blk, keystream_, &aes_key);
        }
        size_t copy_n = std::min((size_t)16 - block_offset_, n - offset);
        for (size_t i = 0; i < copy_n; ++i) {
            out[offset + i] = in[offset + i] ^ keystream_[block_offset_ + i];
        }
        offset += copy_n;
        block_offset_ += copy_n;
        if (block_offset_ == 16) {
            block_offset_ = 0;
            ++counter_;
        }
    }
}

#else

void AesCrypto::ctr_update(const uint8_t* in, uint8_t* out, size_t n) {
    // Stub：无 OpenSSL 时做 XOR 伪加密（仅开发调试，不能用于生产！）
    static bool warned = false;
    if (!warned) {
        LOGW("AES", "[STUB] OpenSSL not linked, using XOR placeholder "
                    "(NOT secure for production)");
        warned = true;
    }
    for (size_t i = 0; i < n; ++i) {
        out[i] = in[i] ^ key_[i % kKeyBytes];
    }
}

#endif

void AesCrypto::encrypt(const uint8_t* in, size_t n, std::vector<uint8_t>& out) {
    if (!ready_ || n == 0) {
        out.assign(in, in + n);
        return;
    }
    out.resize(n);
    std::lock_guard<std::mutex> lock(mtx_);
    ctr_update(in, out.data(), n);
}

}  // namespace dms
