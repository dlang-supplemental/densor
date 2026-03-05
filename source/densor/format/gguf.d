module densor.format.gguf;

import std.mmfile;
import std.system : Endian;
import std.bitmanip;
import std.exception : enforce;
import std.conv : to;
import std.string : representation;

enum GGUFMagic = 0x46554747;

enum GGUFType : uint
{
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9, // Add others as needed
}

enum GGUFMetadataValueType : uint
{
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
}

struct GGUFHeader
{
    uint magic;
    uint version_;
    ulong tensorCount;
    ulong metadataCount;
}

struct GGUFTensorInfo
{
    string name;
    ulong[] dims;
    GGUFType type;
    ulong offset;
}

class GGUFFile
{
    private MmFile mmfile;
    private const(ubyte)[] data;
    public GGUFHeader header;
    public GGUFTensorInfo[] tensorInfos;

    this(string path)
    {
        mmfile = new MmFile(path, MmFile.Mode.read, 0, null);
        data = cast(const(ubyte)[]) mmfile[]; // Slice the whole file
        parse();
    }

    private void parse()
    {
        size_t offset = 0;

        // Read Header
        header.magic = readVal!uint(offset);
        enforce(header.magic == GGUFMagic, "Invalid GGUF magic: " ~ to!string(header.magic));

        header.version_ = readVal!uint(offset);
        header.tensorCount = readVal!ulong(offset);
        header.metadataCount = readVal!ulong(offset);

        // Parse Metadata
        for (ulong i = 0; i < header.metadataCount; ++i)
        {
            string key = readString(offset);
            GGUFMetadataValueType type = cast(GGUFMetadataValueType) readVal!uint(offset);
            skipValue(type, offset);
        }

        // Parse Tensor Infos
        tensorInfos.reserve(cast(size_t) header.tensorCount);
        for (ulong i = 0; i < header.tensorCount; ++i)
        {
            GGUFTensorInfo info;
            info.name = readString(offset);

            uint n_dims = readVal!uint(offset);
            info.dims = new ulong[n_dims];
            for (uint j = 0; j < n_dims; ++j)
            {
                info.dims[j] = readVal!ulong(offset);
            }

            info.type = cast(GGUFType) readVal!uint(offset); // GGUF type is usually uint32?
            // Wait, usually type is GGUFType (uint32). Correct.
            info.offset = readVal!ulong(offset);

            tensorInfos ~= info;
        }

        // Note: info.offset is relative to the base of tensor data, which implies we need to know where tensor data starts.
        // In GGUF v3/v2, tensor info is followed by padding to alignment, then data starts.
        // Or info.offset is absolute? 
        // GGUF spec: "The offset is relative to the start of the data block, which is aligned."
        // We need to calculate data start.
        // The data block starts immediately after the tensor info block, aligned to `alignment`.
        // `alignment` is in metadata "general.alignment", default 32.
        // We need to parse metadata to get alignment.
        // But for now, we just parsed the structure.
    }

    private T readVal(T)(ref size_t offset)
    {
        // Ensure bounds
        enforce(offset + T.sizeof <= data.length, "Unexpected EOF reading " ~ T.stringof);

        // Use peek to read value
        // std.bitmanip.peek takes a pointer to index
        return data.peek!(T, Endian.littleEndian)(&offset);
    }

    private string readString(ref size_t offset)
    {
        ulong len = readVal!ulong(offset); // GGUF strings are 64-bit length prefixed
        enforce(offset + len <= data.length, "String length out of bounds");

        string s = cast(string) data[offset .. offset + cast(size_t) len].idup;
        offset += cast(size_t) len;
        return s;
    }

    private void skipValue(GGUFMetadataValueType type, ref size_t offset)
    {
        final switch (type)
        {
        case GGUFMetadataValueType.UINT8:
        case GGUFMetadataValueType.INT8:
        case GGUFMetadataValueType.BOOL:
            offset += 1;
            break;
        case GGUFMetadataValueType.UINT16:
        case GGUFMetadataValueType.INT16:
            offset += 2;
            break;
        case GGUFMetadataValueType.UINT32:
        case GGUFMetadataValueType.INT32:
        case GGUFMetadataValueType.FLOAT32:
            offset += 4;
            break;
        case GGUFMetadataValueType.UINT64:
        case GGUFMetadataValueType.INT64:
        case GGUFMetadataValueType.FLOAT64:
            offset += 8;
            break;
        case GGUFMetadataValueType.STRING:
            readString(offset);
            break;
        case GGUFMetadataValueType.ARRAY:
            GGUFMetadataValueType itemType = cast(GGUFMetadataValueType) readVal!uint(offset);
            ulong len = readVal!ulong(offset);
            for (ulong i = 0; i < len; ++i)
                skipValue(itemType, offset);
            break;
        }
    }
}
