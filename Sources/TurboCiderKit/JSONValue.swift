import Foundation

/// A Codable JSON value used for namespaced engine-specific options.
public enum TCJSONValue: Codable, Equatable, Sendable {
    case string(String)
    case integer(Int)
    case number(Double)
    case boolean(Bool)
    case object([String: TCJSONValue])
    case array([TCJSONValue])
    case null

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .boolean(value)
        } else if let value = try? container.decode(Int.self) {
            self = .integer(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([String: TCJSONValue].self) {
            self = .object(value)
        } else if let value = try? container.decode([TCJSONValue].self) {
            self = .array(value)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Unsupported JSON value",
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value): try container.encode(value)
        case .integer(let value): try container.encode(value)
        case .number(let value): try container.encode(value)
        case .boolean(let value): try container.encode(value)
        case .object(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .null: try container.encodeNil()
        }
    }
}

extension TCJSONValue: ExpressibleByStringLiteral {
    public init(stringLiteral value: String) { self = .string(value) }
}

extension TCJSONValue: ExpressibleByIntegerLiteral {
    public init(integerLiteral value: Int) { self = .integer(value) }
}

extension TCJSONValue: ExpressibleByFloatLiteral {
    public init(floatLiteral value: Double) { self = .number(value) }
}

extension TCJSONValue: ExpressibleByBooleanLiteral {
    public init(booleanLiteral value: Bool) { self = .boolean(value) }
}

extension TCJSONValue: ExpressibleByArrayLiteral {
    public init(arrayLiteral elements: TCJSONValue...) { self = .array(elements) }
}

extension TCJSONValue: ExpressibleByDictionaryLiteral {
    public init(dictionaryLiteral elements: (String, TCJSONValue)...) {
        self = .object(Dictionary(uniqueKeysWithValues: elements))
    }
}

extension TCJSONValue: ExpressibleByNilLiteral {
    public init(nilLiteral: ()) { self = .null }
}
