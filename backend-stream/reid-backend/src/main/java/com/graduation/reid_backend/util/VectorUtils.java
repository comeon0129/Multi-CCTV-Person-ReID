package com.graduation.reid_backend.util;

public class VectorUtils{
    // 두 벡터 간의 코사인 거리(Cosine Distance) 계산 (1 - Cosine Similarity)
    public static float calculateCosineDistance(float[] v1, float[] v2){
        //1. 두 벡터의 차원이 다르면 비교 자체가 불가능해서 최대 거리 반환
        if (v1.length != v2.length) return 1.0f;

        float dotProduct = 0.0f; //내적
        float normA = 0.0f; //벡터A의 크기
        float normB = 0.0f; //벡터B의 크기

        //2. 내적값과 크기값 계산
        for (int i = 0; i < v1.length; i++) {
            dotProduct += v1[i] * v2[i];
            normA += (float) Math.pow(v1[i], 2);
            normB += (float) Math.pow(v2[i], 2);
        }
        //3. Divide by Zero 방지
        if (normA == 0 || normB == 0) return 1.0f;

        //4. 코사인 유사도 계산(값이 1.0에 가까울 수록 두 벡터가 유사하다는 뜻)
        float similarity = (float) (dotProduct / (Math.sqrt(normA) * Math.sqrt(normB)));

        // 거리 반환 (유사도가 1에 가까울수록 거리는 0에 가까워짐)
        //pgvector의 <=>연산자는 유사도가 아닌 거리를 반환하기에
        //기존 시스템의 threshold 값 기준을 맞추기 위해 1에서 빼준다
        return 1.0f - similarity;
    }
}