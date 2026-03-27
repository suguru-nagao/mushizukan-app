import { useState, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  ScrollView,
  SafeAreaView,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
} from 'react-native';
import { chatWithBugCharacter, ChatCharacter } from '../../lib/gemini';

type Message = {
  role: 'user' | 'assistant';
  text: string;
};

const QUICK_QUESTIONS = [
  'カブトムシはどこにいるの？',
  'なぜ蛍は光るの？',
  'チョウはなぜきれいなの？',
  'アリはなぜ行列するの？',
  'セミはなぜ鳴くの？',
];

const CHARACTERS: { id: ChatCharacter; label: string; emoji: string; desc: string }[] = [
  { id: 'doctor', label: '虫博士',         emoji: '🔬', desc: '丁寧でくわしい' },
  { id: 'friend', label: 'むしむしフレンド', emoji: '🐛', desc: '元気でたのしい' },
];

export default function ChatScreen() {
  const [character, setCharacter] = useState<ChatCharacter>('doctor');
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<ScrollView>(null);

  async function sendMessage(text: string) {
    if (!text.trim() || loading) return;
    const userMsg: Message = { role: 'user', text: text.trim() };
    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    setInput('');
    setLoading(true);

    try {
      const history = messages.map((m) => ({ role: m.role, text: m.text }));
      const reply = await chatWithBugCharacter(text.trim(), character, history);
      setMessages([...newMessages, { role: 'assistant', text: reply || 'うまく答えられなかったよ。もう一度聞いてね！' }]);
    } catch {
      setMessages([...newMessages, { role: 'assistant', text: 'エラーが発生したよ。もう一度試してね！' }]);
    } finally {
      setLoading(false);
      setTimeout(() => scrollRef.current?.scrollToEnd({ animated: true }), 100);
    }
  }

  const currentChar = CHARACTERS.find((c) => c.id === character)!;

  return (
    <SafeAreaView style={styles.root}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>💬 虫博士チャット</Text>
      </View>

      {/* Character selector */}
      <View style={styles.charSelector}>
        {CHARACTERS.map((c) => (
          <TouchableOpacity
            key={c.id}
            style={[styles.charBtn, character === c.id && styles.charBtnActive]}
            onPress={() => {
              setCharacter(c.id);
              setMessages([]);
            }}
          >
            <Text style={styles.charEmoji}>{c.emoji}</Text>
            <Text style={[styles.charLabel, character === c.id && styles.charLabelActive]}>
              {c.label}
            </Text>
            <Text style={styles.charDesc}>{c.desc}</Text>
          </TouchableOpacity>
        ))}
      </View>

      <KeyboardAvoidingView
        style={styles.flex}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        keyboardVerticalOffset={0}
      >
        <ScrollView
          ref={scrollRef}
          style={styles.chatArea}
          contentContainerStyle={styles.chatContent}
          onContentSizeChange={() => scrollRef.current?.scrollToEnd({ animated: true })}
        >
          {/* Welcome message */}
          {messages.length === 0 && (
            <View style={styles.welcomeBox}>
              <Text style={styles.welcomeEmoji}>{currentChar.emoji}</Text>
              <Text style={styles.welcomeText}>
                {character === 'doctor'
                  ? `こんにちは！わたしは虫博士ですよ。\n虫のことなら何でも聞いてくださいなのです！`
                  : `やあ！むしむしフレンドだよ！\n虫のことなんでも聞いてね！🐛`}
              </Text>
            </View>
          )}

          {/* Messages */}
          {messages.map((msg, i) => (
            <View
              key={i}
              style={[
                styles.bubble,
                msg.role === 'user' ? styles.bubbleUser : styles.bubbleAssistant,
              ]}
            >
              {msg.role === 'assistant' && (
                <Text style={styles.bubbleCharEmoji}>{currentChar.emoji}</Text>
              )}
              <View
                style={[
                  styles.bubbleTextWrap,
                  msg.role === 'user' ? styles.bubbleTextWrapUser : styles.bubbleTextWrapAssistant,
                ]}
              >
                <Text style={[styles.bubbleText, msg.role === 'user' && styles.bubbleTextUser]}>
                  {msg.text}
                </Text>
              </View>
            </View>
          ))}

          {loading && (
            <View style={[styles.bubble, styles.bubbleAssistant]}>
              <Text style={styles.bubbleCharEmoji}>{currentChar.emoji}</Text>
              <View style={styles.bubbleTextWrapAssistant}>
                <ActivityIndicator color="#16a34a" size="small" />
              </View>
            </View>
          )}

          {/* Quick questions */}
          {messages.length === 0 && (
            <View style={styles.quickSection}>
              <Text style={styles.quickTitle}>よくある質問</Text>
              {QUICK_QUESTIONS.map((q) => (
                <TouchableOpacity
                  key={q}
                  style={styles.quickCard}
                  onPress={() => sendMessage(q)}
                >
                  <Text style={styles.quickText}>{q}</Text>
                  <Text style={styles.quickArrow}>›</Text>
                </TouchableOpacity>
              ))}
            </View>
          )}
        </ScrollView>

        {/* Input area */}
        <View style={styles.inputArea}>
          <TextInput
            style={styles.input}
            value={input}
            onChangeText={(t) => setInput(t.slice(0, 200))}
            placeholder="虫のことを聞いてみよう..."
            placeholderTextColor="#9ca3af"
            multiline
            maxLength={200}
          />
          <View style={styles.inputRight}>
            <Text style={styles.charCount}>{input.length}/200</Text>
            <TouchableOpacity
              style={[styles.sendBtn, (!input.trim() || loading) && styles.sendBtnDisabled]}
              onPress={() => sendMessage(input)}
              disabled={!input.trim() || loading}
            >
              <Text style={styles.sendBtnText}>送信</Text>
            </TouchableOpacity>
          </View>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#f0fdf4' },
  flex: { flex: 1 },
  header: {
    paddingHorizontal: 20, paddingTop: 16, paddingBottom: 8,
    backgroundColor: '#fff', borderBottomWidth: 1, borderBottomColor: '#e5e7eb',
  },
  headerTitle: { fontSize: 20, fontWeight: 'bold', color: '#14532d' },
  charSelector: {
    flexDirection: 'row', gap: 10, padding: 12,
    backgroundColor: '#fff', borderBottomWidth: 1, borderBottomColor: '#e5e7eb',
  },
  charBtn: {
    flex: 1, alignItems: 'center', paddingVertical: 10, borderRadius: 12,
    backgroundColor: '#f3f4f6', borderWidth: 2, borderColor: 'transparent',
  },
  charBtnActive: { backgroundColor: '#dcfce7', borderColor: '#16a34a' },
  charEmoji: { fontSize: 24, marginBottom: 2 },
  charLabel: { fontSize: 13, fontWeight: 'bold', color: '#6b7280' },
  charLabelActive: { color: '#16a34a' },
  charDesc: { fontSize: 10, color: '#9ca3af', marginTop: 2 },
  chatArea: { flex: 1 },
  chatContent: { padding: 16, paddingBottom: 8 },
  welcomeBox: {
    backgroundColor: '#fff', borderRadius: 16, padding: 20,
    alignItems: 'center', marginBottom: 16,
    shadowColor: '#000', shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.06, shadowRadius: 4, elevation: 2,
  },
  welcomeEmoji: { fontSize: 48, marginBottom: 8 },
  welcomeText: { fontSize: 15, color: '#374151', textAlign: 'center', lineHeight: 22 },
  bubble: { flexDirection: 'row', marginBottom: 12, alignItems: 'flex-end' },
  bubbleUser: { justifyContent: 'flex-end' },
  bubbleAssistant: { justifyContent: 'flex-start' },
  bubbleCharEmoji: { fontSize: 24, marginRight: 6, marginBottom: 4 },
  bubbleTextWrap: { maxWidth: '75%', padding: 12, borderRadius: 16 },
  bubbleTextWrapUser: { backgroundColor: '#16a34a', borderBottomRightRadius: 4 },
  bubbleTextWrapAssistant: {
    backgroundColor: '#fff', borderBottomLeftRadius: 4,
    shadowColor: '#000', shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.06, shadowRadius: 4, elevation: 2,
  },
  bubbleText: { fontSize: 15, color: '#111827', lineHeight: 22 },
  bubbleTextUser: { color: '#fff' },
  quickSection: { marginTop: 8 },
  quickTitle: { fontSize: 14, fontWeight: 'bold', color: '#166534', marginBottom: 8 },
  quickCard: {
    flexDirection: 'row', alignItems: 'center', backgroundColor: '#fff',
    borderRadius: 12, padding: 14, marginBottom: 8,
    shadowColor: '#000', shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.06, shadowRadius: 4, elevation: 2,
  },
  quickText: { flex: 1, fontSize: 14, color: '#374151' },
  quickArrow: { fontSize: 20, color: '#16a34a', fontWeight: 'bold' },
  inputArea: {
    flexDirection: 'row', padding: 12, backgroundColor: '#fff',
    borderTopWidth: 1, borderTopColor: '#e5e7eb', gap: 8, alignItems: 'flex-end',
  },
  input: {
    flex: 1, backgroundColor: '#f3f4f6', borderRadius: 12,
    paddingHorizontal: 14, paddingVertical: 10, fontSize: 15,
    color: '#111827', maxHeight: 100,
  },
  inputRight: { alignItems: 'flex-end', gap: 4 },
  charCount: { fontSize: 10, color: '#9ca3af' },
  sendBtn: {
    backgroundColor: '#16a34a', borderRadius: 10,
    paddingHorizontal: 16, paddingVertical: 10,
  },
  sendBtnDisabled: { backgroundColor: '#d1d5db' },
  sendBtnText: { color: '#fff', fontWeight: 'bold', fontSize: 14 },
});
