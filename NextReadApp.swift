//
//  NextReadApp.swift
//  Single-file SwiftUI app — paste this over the auto-generated ContentView/App files.
//

import SwiftUI

// ============================================================
// CONFIG — change these
// ============================================================
let API_BASE = "http://127.0.0.1:8000"            // Simulator: this works. Phone: use Mac LAN IP or deployed URL.
let API_KEY  = "changeme"                          // Must match backend NEXTREAD_API_KEY env var

// ============================================================
// MODELS
// ============================================================
struct Recommendation: Codable, Identifiable {
    let book_title: String
    let book_author: String
    let year: Int?
    let blurb_excerpt: String?
    let amazon_url: String?
    let source_url: String?
    let signal_types: [String]?

    var id: String { book_title + (book_author) }
}

struct SearchResponse: Codable {
    let query: String?
    let resolved_to: String?
    let alternatives: [String]?
    let include_casual: Bool?
    let blurbs: [Recommendation]
    let casual_shares: [Recommendation]?
}

// ============================================================
// API
// ============================================================
enum APIError: Error, LocalizedError {
    case badResponse(Int, String)
    case decoding(String)
    var errorDescription: String? {
        switch self {
        case .badResponse(let code, let msg): return "HTTP \(code): \(msg)"
        case .decoding(let m): return "Decode: \(m)"
        }
    }
}

class NextReadAPI {
    static func searchEndorser(name: String, includeCasual: Bool) async throws -> SearchResponse {
        let url = URL(string: "\(API_BASE)/search/endorser")!
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.setValue(API_KEY, forHTTPHeaderField: "x-api-key")
        req.httpBody = try JSONEncoder().encode([
            "name": AnyEncodable(name),
            "include_casual": AnyEncodable(includeCasual),
        ])
        req.timeoutInterval = 180  // first search can take 30-90s

        let (data, response) = try await URLSession.shared.data(for: req)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            let body = String(data: data, encoding: .utf8) ?? ""
            throw APIError.badResponse(code, body)
        }
        do {
            return try JSONDecoder().decode(SearchResponse.self, from: data)
        } catch {
            throw APIError.decoding(error.localizedDescription)
        }
    }
}

// Lets us mix String + Bool in a JSON body
struct AnyEncodable: Encodable {
    let value: Encodable
    init(_ v: Encodable) { self.value = v }
    func encode(to encoder: Encoder) throws { try value.encode(to: encoder) }
}

// ============================================================
// VIEWS
// ============================================================
struct ContentView: View {
    @State private var name: String = ""
    @State private var includeCasual: Bool = false
    @State private var loading: Bool = false
    @State private var response: SearchResponse?
    @State private var errorMessage: String?

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 14) {
                Text("Find books endorsed by people whose taste you trust.")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .padding(.horizontal)

                TextField("Name (e.g. Bradley Hope)", text: $name)
                    .textFieldStyle(.roundedBorder)
                    .autocorrectionDisabled(true)
                    .textInputAutocapitalization(.words)
                    .padding(.horizontal)
                    .onSubmit { runSearch() }

                Toggle("Also show casual shares (tweets, blogs, podcasts)", isOn: $includeCasual)
                    .padding(.horizontal)
                    .font(.subheadline)

                Button(action: runSearch) {
                    if loading {
                        ProgressView().padding(.vertical, 4)
                    } else {
                        Text("Find Books")
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 4)
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(name.trimmingCharacters(in: .whitespaces).isEmpty || loading)
                .padding(.horizontal)

                if let errorMessage {
                    Text(errorMessage)
                        .foregroundColor(.red)
                        .font(.callout)
                        .padding(.horizontal)
                }

                if let response {
                    ResultsView(response: response)
                } else if loading {
                    Text("Searching… first search per name takes 30–90 seconds.")
                        .foregroundColor(.secondary)
                        .font(.caption)
                        .padding(.horizontal)
                    Spacer()
                } else {
                    Spacer()
                }
            }
            .padding(.top)
            .navigationTitle("NextRead")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    private func runSearch() {
        let trimmed = name.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return }
        loading = true
        errorMessage = nil
        response = nil

        Task {
            do {
                let r = try await NextReadAPI.searchEndorser(name: trimmed, includeCasual: includeCasual)
                await MainActor.run {
                    self.response = r
                    self.loading = false
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = error.localizedDescription
                    self.loading = false
                }
            }
        }
    }
}

struct ResultsView: View {
    let response: SearchResponse

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 10) {
                if let resolved = response.resolved_to {
                    Text("Blurbs by \(resolved)")
                        .font(.headline)
                        .padding(.horizontal)
                }

                if response.blurbs.isEmpty {
                    Text("\(response.resolved_to ?? "This person") has not written blurbs for any books, yet.")
                        .foregroundColor(.secondary)
                        .italic()
                        .padding(.horizontal)
                } else {
                    ForEach(Array(response.blurbs.enumerated()), id: \.element.id) { idx, rec in
                        BookCard(index: idx + 1, rec: rec)
                    }
                }

                if let casual = response.casual_shares, !casual.isEmpty {
                    Divider().padding(.vertical, 8)
                    Text("Casual shares")
                        .font(.headline)
                        .padding(.horizontal)
                    ForEach(Array(casual.enumerated()), id: \.element.id) { idx, rec in
                        BookCard(index: idx + 1, rec: rec)
                    }
                }
            }
            .padding(.bottom, 40)
        }
    }
}

struct BookCard: View {
    let index: Int
    let rec: Recommendation

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("\(index). \(rec.book_title)")
                .font(.headline)

            let isJunkAuthor = rec.book_author.lowercased().contains("unknown") || rec.book_author == "—"
            if !isJunkAuthor {
                Text("by \(rec.book_author)" + (rec.year.map { " (\($0))" } ?? ""))
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }

            if let signals = rec.signal_types, !signals.isEmpty {
                Text(signals.map { humanLabel($0) }.joined(separator: " · "))
                    .font(.caption)
                    .foregroundColor(.accentColor)
            }

            if let line = rec.blurb_excerpt, !line.isEmpty {
                Text("\u{201C}\(line)\u{201D}")
                    .font(.footnote)
                    .italic()
                    .foregroundColor(.primary.opacity(0.8))
                    .padding(.top, 2)
            }

            HStack(spacing: 12) {
                if let s = rec.amazon_url, let url = URL(string: s) {
                    Link("Buy on Amazon", destination: url)
                        .font(.caption)
                }
                if let s = rec.source_url, !s.isEmpty, let url = URL(string: s) {
                    Link("source", destination: url)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            .padding(.top, 4)
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .cornerRadius(10)
        .padding(.horizontal)
    }

    private func humanLabel(_ sig: String) -> String {
        let map: [String: String] = [
            "blurb": "Back-cover blurb",
            "foreword": "Wrote foreword",
            "introduction": "Wrote introduction",
            "jacket_quote": "Jacket quote",
            "praise_page": "Praise page",
            "tweet": "Tweet",
            "blog_post": "Blog post",
            "substack": "Substack",
            "instagram": "Instagram",
            "podcast_moment": "Podcast moment",
            "interview_moment": "Interview moment",
            "social_post": "Social post",
        ]
        return map[sig] ?? sig.replacingOccurrences(of: "_", with: " ").capitalized
    }
}

@main
struct NextReadApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

#Preview {
    ContentView()
}